from rest_framework import serializers
from .models import CustomUser  # 수정한 사용자 모델

class UserRegistrationSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['email', 'password','nickname']
        extra_kwargs = {
            'password': {'write_only': True},
            'nickname': {'required': False},  # nickname 필드를 필수로 요구하지 않도록 설정
        }

    def create(self, validated_data):
         # nickname이 없을 경우 None을 기본값으로 설정
        nickname = validated_data.get('nickname', None)

        # 'created'는 자동으로 생성되므로 여기서 다루지 않음
        user = CustomUser.objects.create_user(
            email=validated_data['email'],
            password=validated_data['password'],
            nickname=nickname,  # nickname이 없으면 None을 저장
        )
        return user
