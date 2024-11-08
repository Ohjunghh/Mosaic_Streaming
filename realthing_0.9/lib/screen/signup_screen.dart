import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../constants/ip.dart';
class SignUpScreen extends StatefulWidget {
  @override
  _SignUpScreenState createState() => _SignUpScreenState();
}

class _SignUpScreenState extends State<SignUpScreen> {
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _confirmPasswordController = TextEditingController();

  @override
  void dispose() {
    // Dispose of the controllers when the widget is removed from the widget tree.
    _emailController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    super.dispose();
  }

  bool _isValidEmail(String email) {
    final emailRegex = RegExp(r'^[^@]+@[^@]+\.[^@]+$');
    return emailRegex.hasMatch(email);
  }

  bool _isValidPassword(String password) {
    // Update the regex to allow for uppercase and lowercase letters
    final passwordExpert = RegExp(r'^(?=.*[a-zA-Z])(?=.*\d)(?=.*[\W_]).{8,}$');
    return passwordExpert.hasMatch(password);
  }

  bool _isPasswordLengthValid(String password) {
    return password.length < 21; // 비밀번호 길이 체크
  }

  Future<void> _signup(BuildContext context) async {
    final String email = _emailController.text;
    final String password = _passwordController.text;
    final String confirmPassword = _confirmPasswordController.text;

    if (!_isValidEmail(email)) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('잘못된 이메일 형식입니다.')),
      );
      return;
    }

    if (password != confirmPassword) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('비밀번호가 일치하지 않습니다.')),
      );
      return;
    }

    if (!_isValidPassword(password)) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('비밀번호는 최소 8자 이상이며, 영문, 숫자, 특수문자가 조합되어야 합니다.')),
      );
      return;
    }

    if (!_isPasswordLengthValid(password)) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('입력한 비밀번호가 너무 깁니다.')),
      );
      return;
    }

    try {
      print('Sending signup request...');
      final response = await http.post(
        Uri.parse('${ApiConstants.baseUrl}/api/users/signup/'), // Use 10.0.2.2 for emulator
        headers: <String, String>{
          'Content-Type': 'application/json; charset=UTF-8',
        },
        body: jsonEncode(<String, String>{
          'email': email,
          'password': password,
        }),
      );

      print('Response status: ${response.statusCode}');
      print('Response body: ${response.body}');

      if (response.statusCode == 201) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Sign up successful!')),
        );
        Navigator.pop(context);
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Sign up failed. Please try again.')),
        );
      }
    } catch (e) {
      print('Error occurred: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('An error occurred. Please try again later.')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Sign Up'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            // Email Input Field
            TextField(
              controller: _emailController,
              decoration: InputDecoration(
                labelText: 'Email',
                border: OutlineInputBorder(),
                suffixIcon: _isValidEmail(_emailController.text)
                    ? Icon(Icons.check_circle, color: Colors.green)
                    : _emailController.text.isNotEmpty
                    ? Icon(Icons.error, color: Colors.red)
                    : null,
              ),
            ),
            SizedBox(height: 16),
            // Password Input Field
            TextField(
              controller: _passwordController,
              decoration: InputDecoration(
                labelText: 'Password',
                border: OutlineInputBorder(),
                suffixIcon: _isValidPassword(_passwordController.text)
                    ? Icon(Icons.check_circle, color: Colors.green)
                    : _passwordController.text.isNotEmpty
                    ? Icon(Icons.error, color: Colors.red)
                    : null,
              ),
              obscureText: true,
            ),
            SizedBox(height: 16),
            // Confirm Password Input Field
            TextField(
              controller: _confirmPasswordController,
              decoration: InputDecoration(
                labelText: 'Confirm Password',
                border: OutlineInputBorder(),
                errorText: _passwordController.text != _confirmPasswordController.text && _confirmPasswordController.text.isNotEmpty
                    ? 'Passwords do not match'
                    : null,
              ),
              obscureText: true,
            ),
            SizedBox(height: 24),
            ElevatedButton(
              onPressed: () => _signup(context),
              child: Text('Sign Up'),
            ),
          ],
        ),
      ),
    );
  }
}
