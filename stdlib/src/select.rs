// spell-checker:disable

use crate::vm::{
    PyObject, PyObjectRef, PyRef, PyResult, TryFromObject, VirtualMachine, builtins::PyListRef,
    builtins::PyModule,
};
use std::{io, mem};

pub(crate) fn make_module(vm: &VirtualMachine) -> PyRef<PyModule> {
    #[cfg(windows)]
    crate::vm::windows::init_winsock();

    #[cfg(unix)]
    {
        use crate::vm::class::PyClassImpl;
        decl::poll::PyPoll::make_class(&vm.ctx);
    }

    decl::make_module(vm)
}

#[cfg(unix)]
mod platform {
    pub use libc::{FD_ISSET, FD_SET, FD_SETSIZE, FD_ZERO, fd_set, select, timeval};
    pub use std::os::unix::io::RawFd;

    pub const fn check_err(x: i32) -> bool {
        x < 0
    }
}

#[allow(non_snake_case)]
#[cfg(windows)]
mod platform {
    pub use WinSock::{FD_SET as fd_set, FD_SETSIZE, SOCKET as RawFd, TIMEVAL as timeval, select};
    use windows_sys::Win32::Networking::WinSock;

    // based off winsock2.h: https://gist.github.com/piscisaureus/906386#file-winsock2-h-L128-L141

    pub unsafe fn FD_SET(fd: RawFd, set: *mut fd_set) {
        unsafe {
            let mut slot = (&raw mut (*set).fd_array).cast::<RawFd>();
            let fd_count = (*set).fd_count;
            for _ in 0..fd_count {
                if *slot == fd {
                    return;
                }
                slot = slot.add(1);
            }
            // slot == &fd_array[fd_count] at this point
            if fd_count < FD_SETSIZE {
                *slot = fd as RawFd;
                (*set).fd_count += 1;
            }
        }
    }

    pub unsafe fn FD_ZERO(set: *mut fd_set) {
        unsafe { (*set).fd_count = 0 };
    }

    pub unsafe fn FD_ISSET(fd: RawFd, set: *mut fd_set) -> bool {
        use WinSock::__WSAFDIsSet;
        unsafe { __WSAFDIsSet(fd as _, set) != 0 }
    }

    pub fn check_err(x: i32) -> bool {
        x == WinSock::SOCKET_ERROR
    }
}

#[cfg(target_os = "wasi")]
mod platform {
    pub use libc::{FD_SETSIZE, timeval};
    pub use std::os::wasi::io::RawFd;

    pub fn check_err(x: i32) -> bool {
        x < 0
    }

    #[repr(C)]
    pub struct fd_set {
        __nfds: usize,
        __fds: [libc::c_int; FD_SETSIZE],
    }

    #[allow(non_snake_case)]
    pub unsafe fn FD_ISSET(fd: RawFd, set: *const fd_set) -> bool {
        let set = unsafe { &*set };
        let n = set.__nfds;
        for p in &set.__fds[..n] {
            if *p == fd {
                return true;
            }
        }
        false
    }

    #[allow(non_snake_case)]
    pub unsafe fn FD_SET(fd: RawFd, set: *mut fd_set) {
        let set = unsafe { &mut *set };
        let n = set.__nfds;
        for p in &set.__fds[..n] {
            if *p == fd {
                return;
            }
        }
        set.__nfds = n + 1;
        set.__fds[n] = fd;
    }

    #[allow(non_snake_case)]
    pub unsafe fn FD_ZERO(set: *mut fd_set) {
        let set = unsafe { &mut *set };
        set.__nfds = 0;
    }

    unsafe extern "C" {
        pub fn select(
            nfds: libc::c_int,
            readfds: *mut fd_set,
            writefds: *mut fd_set,
            errorfds: *mut fd_set,
            timeout: *const timeval,
        ) -> libc::c_int;
    }
}

use platform::RawFd;
pub use platform::timeval;

#[derive(Traverse)]
struct Selectable {
    obj: PyObjectRef,
    #[pytraverse(skip)]
    fno: RawFd,
}

impl TryFromObject for Selectable {
    fn try_from_object(vm: &VirtualMachine, obj: PyObjectRef) -> PyResult<Self> {
        let fno = obj.try_to_value(vm).or_else(|_| {
            let meth = vm.get_method_or_type_error(
                obj.clone(),
                vm.ctx.interned_str("fileno").unwrap(),
                || "select arg must be an int or object with a fileno() method".to_owned(),
            )?;
            meth.call((), vm)?.try_into_value(vm)
        })?;
        Ok(Self { obj, fno })
    }
}

// Keep it in a MaybeUninit, since on windows FD_ZERO doesn't actually zero the whole thing
#[repr(transparent)]
pub struct FdSet(mem::MaybeUninit<platform::fd_set>);

impl FdSet {
    pub fn new() -> Self {
        // it's just ints, and all the code that's actually
        // interacting with it is in C, so it's safe to zero
        let mut fdset = std::mem::MaybeUninit::zeroed();
        unsafe { platform::FD_ZERO(fdset.as_mut_ptr()) };
        Self(fdset)
    }

    pub fn insert(&mut self, fd: RawFd) {
        unsafe { platform::FD_SET(fd, self.0.as_mut_ptr()) };
    }

    pub fn contains(&mut self, fd: RawFd) -> bool {
        unsafe { platform::FD_ISSET(fd, self.0.as_mut_ptr()) }
    }

    pub fn clear(&mut self) {
        unsafe { platform::FD_ZERO(self.0.as_mut_ptr()) };
    }

    pub fn highest(&mut self) -> Option<RawFd> {
        (0..platform::FD_SETSIZE as RawFd)
            .rev()
            .find(|&i| self.contains(i))
    }
}

pub fn select(
    nfds: libc::c_int,
    readfds: &mut FdSet,
    writefds: &mut FdSet,
    errfds: &mut FdSet,
    timeout: Option<&mut timeval>,
) -> io::Result<i32> {
    let timeout = match timeout {
        Some(tv) => tv as *mut timeval,
        None => std::ptr::null_mut(),
    };
    let ret = unsafe {
        platform::select(
            nfds,
            readfds.0.as_mut_ptr(),
            writefds.0.as_mut_ptr(),
            errfds.0.as_mut_ptr(),
            timeout,
        )
    };
    if platform::check_err(ret) {
        Err(io::Error::last_os_error())
    } else {
        Ok(ret)
    }
}

fn sec_to_timeval(sec: f64) -> timeval {
    timeval {
        tv_sec: sec.trunc() as _,
        tv_usec: (sec.fract() * 1e6) as _,
    }
}

#[pymodule(name = "select")]
mod decl {
    use super::*;
    use crate::vm::{
        PyObjectRef, PyResult, VirtualMachine,
        builtins::PyTypeRef,
        convert::ToPyException,
        function::{Either, OptionalOption},
        stdlib::time,
    };

    #[pyattr]
    fn error(vm: &VirtualMachine) -> PyTypeRef {
        vm.ctx.exceptions.os_error.to_owned()
    }

    #[pyfunction]
    fn select(
        rlist: PyObjectRef,
        wlist: PyObjectRef,
        xlist: PyObjectRef,
        timeout: OptionalOption<Either<f64, isize>>,
        vm: &VirtualMachine,
    ) -> PyResult<(PyListRef, PyListRef, PyListRef)> {
        let mut timeout = timeout.flatten().map(|e| match e {
            Either::A(f) => f,
            Either::B(i) => i as f64,
        });
        if let Some(timeout) = timeout {
            if timeout < 0.0 {
                return Err(vm.new_value_error("timeout must be positive"));
            }
        }
        let deadline = timeout.map(|s| time::time(vm).unwrap() + s);

        let seq2set = |list: &PyObject| -> PyResult<(Vec<Selectable>, FdSet)> {
            let v: Vec<Selectable> = list.try_to_value(vm)?;
            let mut fds = FdSet::new();
            for fd in &v {
                fds.insert(fd.fno);
            }
            Ok((v, fds))
        };

        let (rlist, mut r) = seq2set(&rlist)?;
        let (wlist, mut w) = seq2set(&wlist)?;
        let (xlist, mut x) = seq2set(&xlist)?;

        if rlist.is_empty() && wlist.is_empty() && xlist.is_empty() {
            let empty = vm.ctx.new_list(vec![]);
            return Ok((empty.clone(), empty.clone(), empty));
        }

        let nfds: i32 = [&mut r, &mut w, &mut x]
            .iter_mut()
            .filter_map(|set| set.highest())
            .max()
            .map_or(0, |n| n + 1) as _;

        loop {
            let mut tv = timeout.map(sec_to_timeval);
            let res = super::select(nfds, &mut r, &mut w, &mut x, tv.as_mut());

            match res {
                Ok(_) => break,
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(err) => return Err(err.to_pyexception(vm)),
            }

            vm.check_signals()?;

            if let Some(ref mut timeout) = timeout {
                *timeout = deadline.unwrap() - time::time(vm).unwrap();
                if *timeout < 0.0 {
                    r.clear();
                    w.clear();
                    x.clear();
                    break;
                }
                // retry select() if we haven't reached the deadline yet
            }
        }

        let set2list = |list: Vec<Selectable>, mut set: FdSet| {
            vm.ctx.new_list(
                list.into_iter()
                    .filter(|fd| set.contains(fd.fno))
                    .map(|fd| fd.obj)
                    .collect(),
            )
        };

        let rlist = set2list(rlist, r);
        let wlist = set2list(wlist, w);
        let xlist = set2list(xlist, x);

        Ok((rlist, wlist, xlist))
    }

    #[cfg(unix)]
    #[pyfunction]
    fn poll() -> poll::PyPoll {
        poll::PyPoll::default()
    }

    #[cfg(unix)]
    #[pyattr]
    use libc::{POLLERR, POLLHUP, POLLIN, POLLNVAL, POLLOUT, POLLPRI};

    #[cfg(unix)]
    pub(super) mod poll {
        use super::*;
        use crate::vm::{
            AsObject, PyPayload,
            builtins::PyFloat,
            common::lock::PyMutex,
            convert::{IntoPyException, ToPyObject},
            function::OptionalArg,
            stdlib::io::Fildes,
        };
        use libc::pollfd;
        use num_traits::{Signed, ToPrimitive};
        use std::{
            convert::TryFrom,
            time::{Duration, Instant},
        };

        #[derive(Default)]
        pub(super) struct TimeoutArg<const MILLIS: bool>(pub Option<Duration>);

        impl<const MILLIS: bool> TryFromObject for TimeoutArg<MILLIS> {
            fn try_from_object(vm: &VirtualMachine, obj: PyObjectRef) -> PyResult<Self> {
                let timeout = if vm.is_none(&obj) {
                    None
                } else if let Some(float) = obj.downcast_ref::<PyFloat>() {
                    let float = float.to_f64();
                    if float.is_nan() {
                        return Err(vm.new_value_error("Invalid value NaN (not a number)"));
                    }
                    if float.is_sign_negative() {
                        None
                    } else {
                        let secs = if MILLIS { float * 1000.0 } else { float };
                        Some(Duration::from_secs_f64(secs))
                    }
                } else if let Some(int) = obj.try_index_opt(vm).transpose()? {
                    if int.as_bigint().is_negative() {
                        None
                    } else {
                        let n = int
                            .as_bigint()
                            .to_u64()
                            .ok_or_else(|| vm.new_overflow_error("value out of range"))?;
                        Some(if MILLIS {
                            Duration::from_millis(n)
                        } else {
                            Duration::from_secs(n)
                        })
                    }
                } else {
                    return Err(vm.new_type_error(format!(
                        "expected an int or float for duration, got {}",
                        obj.class()
                    )));
                };
                Ok(Self(timeout))
            }
        }

        #[pyclass(module = "select", name = "poll")]
        #[derive(Default, Debug, PyPayload)]
        pub struct PyPoll {
            // keep sorted
            fds: PyMutex<Vec<pollfd>>,
        }

        #[inline]
        fn search(fds: &[pollfd], fd: i32) -> Result<usize, usize> {
            fds.binary_search_by_key(&fd, |pfd| pfd.fd)
        }

        fn insert_fd(fds: &mut Vec<pollfd>, fd: i32, events: i16) {
            match search(fds, fd) {
                Ok(i) => fds[i].events = events,
                Err(i) => fds.insert(
                    i,
                    pollfd {
                        fd,
                        events,
                        revents: 0,
                    },
                ),
            }
        }

        fn get_fd_mut(fds: &mut [pollfd], fd: i32) -> Option<&mut pollfd> {
            search(fds, fd).ok().map(move |i| &mut fds[i])
        }

        fn remove_fd(fds: &mut Vec<pollfd>, fd: i32) -> Option<pollfd> {
            search(fds, fd).ok().map(|i| fds.remove(i))
        }

        // new EventMask type
        #[derive(Copy, Clone)]
        #[repr(transparent)]
        pub struct EventMask(pub i16);

        impl TryFromObject for EventMask {
            fn try_from_object(vm: &VirtualMachine, obj: PyObjectRef) -> PyResult<Self> {
                use crate::builtins::PyInt;
                let int = obj
                    .downcast::<PyInt>()
                    .map_err(|_| vm.new_type_error("argument must be an integer"))?;

                let val = int.as_bigint();
                if val.is_negative() {
                    return Err(vm.new_value_error("negative event mask"));
                }

                // Try converting to i16, should raise OverflowError if too large
                let mask = i16::try_from(val)
                    .map_err(|_| vm.new_overflow_error("event mask value out of range"))?;

                Ok(Self(mask))
            }
        }

        const DEFAULT_EVENTS: i16 = libc::POLLIN | libc::POLLPRI | libc::POLLOUT;

        #[pyclass]
        impl PyPoll {
            #[pymethod]
            fn register(
                &self,
                Fildes(fd): Fildes,
                eventmask: OptionalArg<EventMask>,
            ) -> PyResult<()> {
                let mask = match eventmask {
                    OptionalArg::Present(event_mask) => event_mask.0,
                    OptionalArg::Missing => DEFAULT_EVENTS,
                };
                insert_fd(&mut self.fds.lock(), fd, mask);
                Ok(())
            }

            #[pymethod]
            fn modify(
                &self,
                Fildes(fd): Fildes,
                eventmask: EventMask,
                vm: &VirtualMachine,
            ) -> PyResult<()> {
                let mut fds = self.fds.lock();
                // CPython raises KeyError if fd is not registered, match that behavior
                let pfd = get_fd_mut(&mut fds, fd)
                    .ok_or_else(|| vm.new_key_error(vm.ctx.new_int(fd).into()))?;
                pfd.events = eventmask.0;
                Ok(())
            }

            #[pymethod]
            fn unregister(&self, Fildes(fd): Fildes, vm: &VirtualMachine) -> PyResult<()> {
                let removed = remove_fd(&mut self.fds.lock(), fd);
                removed
                    .map(drop)
                    .ok_or_else(|| vm.new_key_error(vm.ctx.new_int(fd).into()))
            }

            #[pymethod]
            fn poll(
                &self,
                timeout: OptionalArg<TimeoutArg<true>>,
                vm: &VirtualMachine,
            ) -> PyResult<Vec<PyObjectRef>> {
                let mut fds = self.fds.lock();
                let TimeoutArg(timeout) = timeout.unwrap_or_default();
                let timeout_ms = match timeout {
                    Some(d) => i32::try_from(d.as_millis())
                        .map_err(|_| vm.new_overflow_error("value out of range"))?,
                    None => -1i32,
                };
                let deadline = timeout.map(|d| Instant::now() + d);
                let mut poll_timeout = timeout_ms;
                loop {
                    let res = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as _, poll_timeout) };
                    match nix::Error::result(res) {
                        Ok(_) => break,
                        Err(nix::Error::EINTR) => vm.check_signals()?,
                        Err(e) => return Err(e.into_pyexception(vm)),
                    }
                    if let Some(d) = deadline {
                        if let Some(remaining) = d.checked_duration_since(Instant::now()) {
                            poll_timeout = remaining.as_millis() as i32;
                        } else {
                            break;
                        }
                    }
                }
                Ok(fds
                    .iter()
                    .filter(|pfd| pfd.revents != 0)
                    .map(|pfd| (pfd.fd, pfd.revents & 0xfff).to_pyobject(vm))
                    .collect())
            }
        }
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_os = "redox"))]
    #[pyattr(name = "epoll", once)]
    fn epoll(vm: &VirtualMachine) -> PyTypeRef {
        use crate::vm::class::PyClassImpl;
        epoll::PyEpoll::make_class(&vm.ctx)
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_os = "redox"))]
    #[pyattr]
    use libc::{
        EPOLL_CLOEXEC, EPOLLERR, EPOLLEXCLUSIVE, EPOLLHUP, EPOLLIN, EPOLLMSG, EPOLLONESHOT,
        EPOLLOUT, EPOLLPRI, EPOLLRDBAND, EPOLLRDHUP, EPOLLRDNORM, EPOLLWAKEUP, EPOLLWRBAND,
        EPOLLWRNORM,
    };
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "redox"))]
    #[pyattr]
    const EPOLLET: u32 = libc::EPOLLET as u32;

    #[cfg(any(target_os = "linux", target_os = "android", target_os = "redox"))]
    pub(super) mod epoll {
        use super::*;
        use crate::vm::{
            PyPayload,
            builtins::PyTypeRef,
            common::lock::{PyRwLock, PyRwLockReadGuard},
            convert::{IntoPyException, ToPyObject},
            function::OptionalArg,
            stdlib::io::Fildes,
            types::Constructor,
        };
        use rustix::event::epoll::{self, EventData, EventFlags};
        use std::ops::Deref;
        use std::os::fd::{AsRawFd, IntoRawFd, OwnedFd};
        use std::time::Instant;

        #[pyclass(module = "select", name = "epoll")]
        #[derive(Debug, rustpython_vm::PyPayload)]
        pub struct PyEpoll {
            epoll_fd: PyRwLock<Option<OwnedFd>>,
        }

        #[derive(FromArgs)]
        pub struct EpollNewArgs {
            #[pyarg(any, default = -1)]
            sizehint: i32,
            #[pyarg(any, default = 0)]
            flags: i32,
        }

        impl Constructor for PyEpoll {
            type Args = EpollNewArgs;
            fn py_new(cls: PyTypeRef, args: EpollNewArgs, vm: &VirtualMachine) -> PyResult {
                if let ..=-2 | 0 = args.sizehint {
                    return Err(vm.new_value_error("negative sizehint"));
                }
                if !matches!(args.flags, 0 | libc::EPOLL_CLOEXEC) {
                    return Err(vm.new_os_error("invalid flags".to_owned()));
                }
                Self::new()
                    .map_err(|e| e.into_pyexception(vm))?
                    .into_ref_with_type(vm, cls)
                    .map(Into::into)
            }
        }

        #[derive(FromArgs)]
        struct EpollPollArgs {
            #[pyarg(any, default)]
            timeout: poll::TimeoutArg<false>,
            #[pyarg(any, default = -1)]
            maxevents: i32,
        }

        #[pyclass(with(Constructor))]
        impl PyEpoll {
            fn new() -> std::io::Result<Self> {
                let epoll_fd = epoll::create(epoll::CreateFlags::CLOEXEC)?;
                let epoll_fd = Some(epoll_fd).into();
                Ok(Self { epoll_fd })
            }

            #[pymethod]
            fn close(&self) -> std::io::Result<()> {
                let fd = self.epoll_fd.write().take();
                if let Some(fd) = fd {
                    nix::unistd::close(fd.into_raw_fd())?;
                }
                Ok(())
            }

            #[pygetset]
            fn closed(&self) -> bool {
                self.epoll_fd.read().is_none()
            }

            fn get_epoll(
                &self,
                vm: &VirtualMachine,
            ) -> PyResult<impl Deref<Target = OwnedFd> + '_> {
                PyRwLockReadGuard::try_map(self.epoll_fd.read(), |x| x.as_ref())
                    .map_err(|_| vm.new_value_error("I/O operation on closed epoll object"))
            }

            #[pymethod]
            fn fileno(&self, vm: &VirtualMachine) -> PyResult<i32> {
                self.get_epoll(vm).map(|epoll_fd| epoll_fd.as_raw_fd())
            }

            #[pyclassmethod]
            fn fromfd(cls: PyTypeRef, fd: OwnedFd, vm: &VirtualMachine) -> PyResult<PyRef<Self>> {
                let epoll_fd = Some(fd).into();
                Self { epoll_fd }.into_ref_with_type(vm, cls)
            }

            #[pymethod]
            fn register(
                &self,
                fd: Fildes,
                eventmask: OptionalArg<u32>,
                vm: &VirtualMachine,
            ) -> PyResult<()> {
                let events = match eventmask {
                    OptionalArg::Present(mask) => EventFlags::from_bits_retain(mask),
                    OptionalArg::Missing => EventFlags::IN | EventFlags::PRI | EventFlags::OUT,
                };
                let epoll_fd = &*self.get_epoll(vm)?;
                let data = EventData::new_u64(fd.as_raw_fd() as u64);
                epoll::add(epoll_fd, fd, data, events).map_err(|e| e.into_pyexception(vm))
            }

            #[pymethod]
            fn modify(&self, fd: Fildes, eventmask: u32, vm: &VirtualMachine) -> PyResult<()> {
                let events = EventFlags::from_bits_retain(eventmask);
                let epoll_fd = &*self.get_epoll(vm)?;
                let data = EventData::new_u64(fd.as_raw_fd() as u64);
                epoll::modify(epoll_fd, fd, data, events).map_err(|e| e.into_pyexception(vm))
            }

            #[pymethod]
            fn unregister(&self, fd: Fildes, vm: &VirtualMachine) -> PyResult<()> {
                let epoll_fd = &*self.get_epoll(vm)?;
                epoll::delete(epoll_fd, fd).map_err(|e| e.into_pyexception(vm))
            }

            #[pymethod]
            fn poll(&self, args: EpollPollArgs, vm: &VirtualMachine) -> PyResult<PyListRef> {
                let poll::TimeoutArg(timeout) = args.timeout;
                let maxevents = args.maxevents;

                let mut poll_timeout =
                    timeout
                        .map(rustix::event::Timespec::try_from)
                        .transpose()
                        .map_err(|_| vm.new_overflow_error("timeout is too large"))?;

                let deadline = timeout.map(|d| Instant::now() + d);
                let maxevents = match maxevents {
                    ..-1 => {
                        return Err(vm.new_value_error(format!(
                            "maxevents must be greater than 0, got {maxevents}"
                        )));
                    }
                    -1 => libc::FD_SETSIZE - 1,
                    _ => maxevents as usize,
                };

                let mut events = Vec::<epoll::Event>::with_capacity(maxevents);

                let epoll = &*self.get_epoll(vm)?;

                loop {
                    events.clear();
                    match epoll::wait(
                        epoll,
                        rustix::buffer::spare_capacity(&mut events),
                        poll_timeout.as_ref(),
                    ) {
                        Ok(_) => break,
                        Err(rustix::io::Errno::INTR) => vm.check_signals()?,
                        Err(e) => return Err(e.into_pyexception(vm)),
                    }
                    if let Some(deadline) = deadline {
                        if let Some(new_timeout) = deadline.checked_duration_since(Instant::now()) {
                            poll_timeout = Some(new_timeout.try_into().unwrap());
                        } else {
                            break;
                        }
                    }
                }

                let ret = events
                    .iter()
                    .map(|ev| (ev.data.u64() as i32, { ev.flags }.bits()).to_pyobject(vm))
                    .collect();

                Ok(vm.ctx.new_list(ret))
            }

            #[pymethod]
            fn __enter__(zelf: PyRef<Self>, vm: &VirtualMachine) -> PyResult<PyRef<Self>> {
                zelf.get_epoll(vm)?;
                Ok(zelf)
            }

            #[pymethod]
            fn __exit__(
                &self,
                _exc_type: OptionalArg,
                _exc_value: OptionalArg,
                _exc_tb: OptionalArg,
            ) -> std::io::Result<()> {
                self.close()
            }
        }
    }
}
