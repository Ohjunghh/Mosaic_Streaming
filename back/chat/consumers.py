# import json

# from asgiref.sync import async_to_sync
# from channels.generic.websocket import WebsocketConsumer


# class ChatConsumer(WebsocketConsumer):
#     def connect(self):
#         self.room_name = self.scope["url_route"]["kwargs"]["room_name"]
#         #소비자에게 WebSocket 연결을 연 'room_name'URL 경로에서 매개변수를 가져옵니다 .chat/routing.py
#         #각 소비자는 연결에 대한 정보가 포함된 범위를 가지며 , 특히 URL 경로의 위치 또는 키워드 인수와 현재 인증된 사용자(있는 경우)가 포함됩니다.
        
#         self.room_group_name = f"chat_{self.room_name}"
#         # 따옴표나 이스케이프 없이 사용자가 지정한 방 이름에서 직접 채널 그룹 이름을 구성합니다.
#         # 그룹 이름에는 영숫자, 하이픈, 밑줄 또는 마침표만 포함될 수 있습니다. 따라서 이 예제 코드는 다른 문자가 있는 방 이름에서는 실패합니다.

#         # Join room group
#         async_to_sync(self.channel_layer.group_add)(
#             self.room_group_name, self.channel_name
#         )
#         # 그룹에 가입합니다.
#         # async_to_sync(...)ChatConsumer는 동기 WebsocketConsumer이지만 비동기 채널 계층 메서드를 호출하기 때문에 래퍼가 필요합니다. (모든 채널 계층 메서드는 비동기입니다. )        
#         # 그룹 이름은 ASCII 영숫자, 하이픈, 마침표로만 제한되며 기본 백엔드에서 최대 길이는 100으로 제한됩니다. 이 코드는 방 이름에서 직접 그룹 이름을 구성하므로 방 이름에 그룹 이름에 유효하지 않은 문자가 포함되거나 길이 제한을 초과하면 실패합니다.

#         self.accept()
#         # WebSocket 연결을 허용합니다.
#         # accept()메서드 내에서 호출하지 않으면 connect()연결이 거부되고 닫힙니다. 예를 들어 요청하는 사용자가 요청된 작업을 수행할 권한이 없기 때문에 연결을 거부할 수 있습니다.
#         # 


#     def disconnect(self, close_code):
#         # Leave room group
#         async_to_sync(self.channel_layer.group_discard)(
#             self.room_group_name, self.channel_name
#         )

#     # Receive message from WebSocket
#     def receive(self, text_data):
#         text_data_json = json.loads(text_data)
#         message = text_data_json["message"]

#         # Send message to room group
#         async_to_sync(self.channel_layer.group_send)(
#             self.room_group_name, {"type": "chat.message", "message": message}
#         )

#     # Receive message from room group
#     def chat_message(self, event):
#         message = event["message"]

#         # Send message to WebSocket
#         self.send(text_data=json.dumps({"message": message}))

# # consumers.py 파일을 만드는 이유는 Django Channels를 사용하여 WebSocket 연결을 처리하는 데 필요한 소비자(Consumer)를 정의하기 위해서입니다.

# # WebSocket 연결 처리: Django Channels는 HTTP 이외의 프로토콜인 WebSocket을 처리할 수 있는 기능을 제공합니다. 이를 위해 consumers.py 파일에서는 WebSocket 연결을 수락하고, 연결된 클라이언트와의 상호작용을 담당할 소비자 클래스를 정의합니다.

# # 루팅 설정과의 연동: Django는 HTTP 요청을 받으면 URLconf를 참조하여 해당하는 뷰 함수를 호출합니다. 마찬가지로 Channels는 WebSocket 연결을 받으면 루트 라우팅 설정(routing.py 등)을 참조하여 소비자를 호출합니다. 소비자는 WebSocket에서 발생하는 이벤트를 처리하고, 클라이언트로부터 받은 메시지를 처리하는 로직을 담당합니다.

# # 공통 경로 사용 권장: /ws/와 같은 공통 경로 접두사를 사용하는 것은 WebSocket 연결을 일반 HTTP 연결과 구별하기 위한 좋은 관행입니다. 이를 통해 프로덕션 환경에서 Nginx와 같은 웹 서버를 사용하여 HTTP와 WebSocket 요청을 각각 다른 서버로 라우팅하는 것이 가능해집니다.

# # 배포 전략: 대규모 사이트의 경우, HTTP 요청은 Gunicorn과 같은 WSGI 서버로, WebSocket 요청은 Daphne과 같은 ASGI 서버로 라우팅할 수 있습니다. 이렇게 구성하면 각 서버가 각각의 프로토콜을 효과적으로 처리할 수 있습니다.

# # 요약하자면, consumers.py 파일은 Django Channels를 사용하여 WebSocket을 처리하기 위한 중요한 구성 요소로, WebSocket 연결을 받아들이고 이벤트를 처리하는 소비자를 정의하는 곳입니다.

import json

from channels.generic.websocket import AsyncWebsocketConsumer


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope["url_route"]["kwargs"]["room_name"]
        self.room_group_name = f"chat_{self.room_name}"

        # Join room group
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    # Receive message from WebSocket
    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json["message"]

        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name, {"type": "chat.message", "message": message}
        )

    # Receive message from room group
    async def chat_message(self, event):
        message = event["message"]

        # Send message to WebSocket
        await self.send(text_data=json.dumps({"message": message}))




