from importlib import import_module
from pprint import pprint

try:
    m = import_module('app.main')
    app = getattr(m, 'app', None)
    routes = [r.path for r in app.routes] if app is not None else []
    print('APP_IMPORT_OK')
    pprint(routes)
except Exception as e:
    print('APP_IMPORT_ERROR:', e)
