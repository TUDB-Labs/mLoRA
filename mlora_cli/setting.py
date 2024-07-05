G_PORT = 8000
G_HOST = "http://127.0.0.1"


def url() -> str:
    global G_HOST
    global G_PORT

    return G_HOST + ":" + str(G_PORT)


def help_set():
    print("Usage of a set:")
    print("  host")
    print("    set the host.")
    print("  port")
    print("    set the port.")


def do_set(_, args):
    args = args.split(" ")

    global G_PORT
    global G_HOST

    if args[0] == "host":
        G_HOST = args[1]
    elif args[0] == "port":
        # convert to int to check
        G_PORT = int(args[1])
    else:
        help_set()
