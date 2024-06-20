G_PORT = 8000
G_HOST = "http://127.0.0.1"


def url() -> str:
    global G_HOST
    global G_PORT

    return G_HOST + ":" + str(G_PORT)
