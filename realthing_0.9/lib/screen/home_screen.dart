//홈스크린

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import 'account_screen.dart';
import 'register_face_screen.dart';
import 'live_stream_home_screen.dart';
import 'my_video_screen.dart';
import 'vlc_player_screen.dart';
import '../constants/ip.dart';

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 1;
  String? _profileImageUrl;

  @override
  void initState() {
    super.initState();
    _loadUserProfile();  // 프로필 로드 함수 호출
  }

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  Future<void> _loadUserProfile() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    String? token = prefs.getString('token');
    if (token == null) {
      print("No token found. Exiting function.");
      return;  // 토큰이 없으면 리턴
    }

    final response = await http.get(
      Uri.parse('${ApiConstants.baseUrl}/api/users/profile'),
      headers: {
        'Authorization': 'Bearer $token',
      },
    );
    // 응답 코드 출력
    print("Response status code: ${response.statusCode}");
    print("Response body: ${response.body}");

    if (response.statusCode == 200) {
      var data = jsonDecode(response.body);

      setState(() {
        // 서버에서 프로필 이미지가 있으면 해당 URL 사용, 없으면 null 설정
        _profileImageUrl = data['profile_image'] != null ? data['profile_image'] : null;
      });
    } else {
      print('Failed to load profile');
      setState(() {
        _profileImageUrl = null;  // 에러가 발생하면 이미지를 표시하지 않음
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'PrivStream',
          style: TextStyle(color: Colors.pink),
        ),
        backgroundColor: Colors.white,
        centerTitle: true,
      ),
      body: _buildBody(),
      bottomNavigationBar: BottomNavigationBar(
        items: const <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Icon(Icons.video_library),
            label: '동영상',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: '홈',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.live_tv),
            label: '라이브 스트리밍',
          ),
        ],
        currentIndex: _selectedIndex,
        selectedItemColor: Colors.pink,
        onTap: _onItemTapped,
      ),
    );
  }

  Widget _buildBody() {
    switch (_selectedIndex) {
      case 0:
        return MyVideosScreen();
      case 1:
        return _buildHomeContent();
      case 2:
        return LiveStreamScreen();
      default:
        return Center(child: Text('홈 화면'));
    }
  }

  Widget _buildHomeContent() {
    return RefreshIndicator(
      onRefresh: _loadUserProfile,  // 새로고침 시 프로필 로드
      child: ListView(
        children: [
          SizedBox(height: 20),
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                CircleAvatar(
                  radius: 50,
                  backgroundImage: _profileImageUrl != null
                      ? NetworkImage(_profileImageUrl!)  // URL이 있으면 이미지 표시
                      : null,  // URL이 없으면 표시하지 않음
                ),
                SizedBox(height: 20),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20.0),
                  child: Column(
                    children: [
                      Card(
                        elevation: 5,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: SizedBox(
                          width: double.infinity,
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              padding: EdgeInsets.symmetric(vertical: 15),
                              backgroundColor: Colors.pink, // 배경 색상
                            ),
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                    builder: (context) => RegisterFaceScreen()),
                              ).then((result) {
                                if (result == true) {
                                  // 프로필 등록 후 새로고침
                                  _loadUserProfile();
                                }
                              });
                            },
                            child: Text('얼굴 등록하기', style: TextStyle(fontSize: 18)),
                          ),
                        ),
                      ),
                      SizedBox(height: 15), // 버튼 간의 간격
                      Card(
                        elevation: 5,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: SizedBox(
                          width: double.infinity,
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              padding: EdgeInsets.symmetric(vertical: 15),
                              backgroundColor: Colors.pink, // 배경 색상
                            ),
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                    builder: (context) => AccountScreen()),
                              ).then((result) {
                                if (result == true) {
                                  // 계정 관리 후 새로고침
                                  _loadUserProfile();
                                }
                              });
                            },
                            child: Text('계정 관리', style: TextStyle(fontSize: 18)),
                          ),
                        ),
                      ),
                      SizedBox(height: 15), // 버튼 간의 간격 추가
                      // 새로운 버튼 추가
                      Card(
                        elevation: 5,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: SizedBox(
                          width: double.infinity,
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              padding: EdgeInsets.symmetric(vertical: 15),
                              backgroundColor: Colors.pink, // 배경 색상
                            ),
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                    builder: (context) => UrlAndStreamKeyInputPage()),
                              );
                            },
                            child: Text('비디오 플레이어', style: TextStyle(fontSize: 18)),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}