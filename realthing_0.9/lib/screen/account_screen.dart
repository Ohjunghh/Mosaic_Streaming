// 계정 스크린
import 'dart:io'; // 파일 처리를 위한 import
import 'register_face_screen.dart';
import 'my_video_screen.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart'; // 이미지 선택을 위한 import
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart'; // SharedPreferences import
import 'package:http_parser/http_parser.dart'; // MediaType import
import '../constants/ip.dart';
import 'home_screen.dart';

class AccountScreen extends StatefulWidget {
  @override
  _AccountScreenState createState() => _AccountScreenState();
}

class _AccountScreenState extends State<AccountScreen> {
  File? _profileImage; // 선택한 이미지 파일
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        _profileImage = File(pickedFile.path);
      });
    }
  }

  Future<void> _uploadProfileImage() async {
    if (_profileImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Please select an image first.')));
      return;
    }

    SharedPreferences prefs = await SharedPreferences.getInstance();
    String? token = prefs.getString('token');

    if (token == null) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('You need to log in first.')));
      return;
    }

    var request = http.MultipartRequest(
      'POST',
      Uri.parse('${ApiConstants.baseUrl}/api/users/profile/'),
    );
    request.headers['Authorization'] = 'Bearer $token';
    request.files.add(await http.MultipartFile.fromPath(
      'profile_image',
      _profileImage!.path,
      contentType: MediaType('image', 'jpeg'),
    ));

    var response = await request.send();

    if (response.statusCode == 200) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Profile image uploaded successfully.')));
      Navigator.pop(context, true);  // 홈 화면에 성공 여부 전달 (true)
    } else {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed to upload profile image.')));
    }
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Account'),
        backgroundColor: Colors.white,
        centerTitle: true,
      ),
      body: ListView(
        children: [
          ListTile(
            leading: Icon(Icons.person),
            title: Text('Account Information'),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => AccountInfoScreen()),
              );
            },
          ),
          Divider(),
          ListTile(
            leading: Icon(Icons.upload),
            title: Text('Upload Profile Picture'),
            onTap: _pickImage, // 이미지 선택
          ),
          if (_profileImage != null) // 이미지가 선택된 경우 미리 보기 제공
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                children: [
                  Image.file(
                    _profileImage!,
                    height: 150,
                    width: 150,
                  ),
                  SizedBox(height: 20),
                  ElevatedButton(
                    onPressed: _uploadProfileImage, // 업로드 버튼
                    child: Text('Upload Profile Image'),
                  ),
                ],
              ),
            ),
          Divider(),
          ListTile(
            leading: Icon(Icons.video_library),
            title: Text('My Videos'), // 이름 변경
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => MyVideosScreen()), // 새 화면으로 연결
              );
            },
          ),
          Divider(),
          ListTile(
            leading: Icon(Icons.history),
            title: Text('Recently Watched'),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => RecentlyWatchedScreen()),
              );
            },
          ),
          Divider(),
          ListTile(
            leading: Icon(Icons.face),
            title: Text('Register Face'),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => RegisterFaceScreen()),
              );
            },
          ),
          Divider(),
          ListTile(
            leading: Icon(Icons.settings),
            title: Text('Settings'),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => SettingsScreen()),
              );
            },
          ),
          Divider(),
          ListTile(
            leading: Icon(Icons.logout),
            title: Text('Log Out'),
            onTap: () async {
              // 로그아웃 처리
              SharedPreferences prefs = await SharedPreferences.getInstance();
              await prefs.remove('token'); // 토큰 제거
              await prefs.remove('user_email'); // 이메일 제거

              // 로그아웃 후 로그인 화면으로 이동
              Navigator.pushReplacement(
                context,
                MaterialPageRoute(builder: (context) => HomeScreen()),
              ); // 로그인 화면으로 이동
            },
          ),
        ],
      ),
    );
  }
}

// 세부 화면들...

class AccountInfoScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Account Information'),
      ),
      body: Center(child: Text('Account Information')),
    );
  }
}

class RecentlyWatchedScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Recently Watched'),
      ),
      body: Center(child: Text('Recently Watched Videos')),
    );
  }
}

class SettingsScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Settings'),
      ),
      body: Center(child: Text('Settings Content')),
    );
  }
}
