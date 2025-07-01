from holo.notify import sendNotification as __sendNotification

DEFAULT_CHANEL:str = "https://ntfy.sh/__tradingbot_notifsToPhone_184632091"


def sendNotification(
        message:"str|bytes", title:"str|None"=None, priorityLevel:"int|str"="default",
        tags:"str|list[str]|None"=None, chanel:"None|str"=None, raiseReqError:bool=False)->bool:
    """send the notification and return if the request sended correctly\n
    `raiseError` is whether an error will be raise if request fail (will print the status otherwise)"""
    chanel = DEFAULT_CHANEL if chanel is None else chanel
    return __sendNotification(chanel, message, title, priorityLevel, tags, raiseReqError)

