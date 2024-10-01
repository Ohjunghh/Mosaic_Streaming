from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
#AbstractUser
from django.db import models

# Create your models here.
class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('이메일은 필수입니다.')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError('관리자는 is_staff=True 이어야 합니다.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('관리자는 is_superuser=True 이어야 합니다.')

        return self.create_user(email, password, **extra_fields)
    
class CustomUser(AbstractBaseUser):
    email = models.EmailField(unique=True)  # 이메일 필드
    nickname = models.CharField(max_length=50, blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True) # 자동 생성 

    # 이메일을 기본 인증 필드로 설정
    USERNAME_FIELD = 'email'  # 이메일을 기본 인증 필드로 설정
    REQUIRED_FIELDS = []  # 이메일 외에 추가 필수 필드 없음

     # 사용자 매니저 설정
    objects = CustomUserManager()

    def __str__(self):
        return self.email
    
def user_directory_path(instance, filename):
    # 파일이 저장될 경로: media/faces/{user_id}/{filename}
    return f'faces/{instance.user.id}/{filename}'

def profile_directory_path(instance, filename):
    # 프로필 사진이 저장될 경로: media/profile/{user_id}/{filename}
    return f'profile/{instance.user.id}/{filename}'

class UserImage(models.Model):
    user = models.ForeignKey(CustomUser, related_name='images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to=user_directory_path)
    uploaded = models.DateTimeField(auto_now_add=True)

class ProcessedVideo(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)  # 사용자와 연결
    name = models.CharField(max_length=255)  # 비디오 이름
    url = models.URLField()  # 비디오 URL
    created_at = models.DateTimeField(auto_now_add=True)  # 생성 일자
    def __str__(self):
        return self.name

class ProfileImage(models.Model):
    user = models.ForeignKey(CustomUser, related_name='profile', on_delete=models.CASCADE)
    image = models.ImageField(upload_to=profile_directory_path)  # 함수 사용
    uploaded = models.DateTimeField(auto_now_add=True)
    is_primary = models.BooleanField(default=False)  # 기본 프로필 이미지 여부

    def __str__(self):
        return f"{self.user.email}'s profile image"