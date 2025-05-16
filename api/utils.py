def construct_response(data:dict, error:str = None) -> dict:
    """construct a response of unified format"""
    if error is None:
        return {
            'data': data,
            'success': True,
            'message': ''
        }
    else:
        return {
            'data': None,
            'success': False,
            'message': error
        }