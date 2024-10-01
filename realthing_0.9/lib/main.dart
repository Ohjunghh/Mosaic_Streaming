//main.dart
import 'package:flutter/material.dart';
import 'screen/login_screen.dart';
import 'screen/signup_screen.dart';
import 'screen/home_screen.dart'; // 홈 화면 임포트

void main() {
  WidgetsFlutterBinding.ensureInitialized();//이거 추가
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Streaming App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: LoginScreen(),
      routes: {
        '/signup': (context) => SignUpScreen(),
        '/home': (context) => HomeScreen(), // 홈 화면 라우트
      },
    );
  }
}
/*
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'screen/home_screen.dart';
import 'screen/login_screen.dart';
import 'screen/signup_screen.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  Future<bool> _checkLoginStatus() async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString('token');
    return token != null;
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<bool>(
      future: _checkLoginStatus(),
      builder: (context, snapshot) {
        // 로그인 상태를 확인하는 동안 로딩 표시
        if (snapshot.connectionState == ConnectionState.waiting) {
          return MaterialApp(
            home: Scaffold(
              body: Center(child: CircularProgressIndicator()),
            ),
          );
        }

        // 로그인 상태에 따라 초기 화면 설정
        return MaterialApp(
          title: 'Streaming App',
          theme: ThemeData(
            primarySwatch: Colors.blue,
          ),
          home: snapshot.data == true ? HomeScreen() : LoginScreen(),
          routes: {
            '/signup': (context) => SignUpScreen(),
            '/home': (context) => HomeScreen(),
          },
        );
      },
    );
  }
}
*/