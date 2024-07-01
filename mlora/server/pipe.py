import multiprocessing
import multiprocessing.connection

# define pipe to communication the command
# g_m: model side use, g_s: server side use
g_m_dispatcher, g_s_dispatcher = multiprocessing.Pipe(True)
g_m_create_task, g_s_create_task = multiprocessing.Pipe(True)
g_m_notify_terminate_task, g_s_notify_terminate_task = multiprocessing.Pipe(True)


def m_dispatcher() -> multiprocessing.connection.Connection:
    global g_m_dispatcher
    return g_m_dispatcher


def m_create_task() -> multiprocessing.connection.Connection:
    global g_m_create_task
    return g_m_create_task


def m_notify_terminate_task() -> multiprocessing.connection.Connection:
    global g_m_notify_terminate_task
    return g_m_notify_terminate_task
