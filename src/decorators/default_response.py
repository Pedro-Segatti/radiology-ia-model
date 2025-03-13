from flask import jsonify
from functools import wraps

def standard_response(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            data, status_code = f(*args, **kwargs)
            response = {
                'success': True,
                'message': 'Request successful',
                'data': data
            }
            return jsonify(response), status_code
        except Exception as e:
            response = {
                'success': False,
                'message': str(e)
            }
            return jsonify(response), 500
    return decorated_function
