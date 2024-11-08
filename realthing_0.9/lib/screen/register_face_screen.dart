import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart'; // MediaType import
import 'package:mime/mime.dart';
import 'package:shared_preferences/shared_preferences.dart'; // SharedPreferences 추가
import '../constants/ip.dart';

class RegisterFaceScreen extends StatefulWidget {
  @override
  _RegisterFaceScreenState createState() => _RegisterFaceScreenState();
}

class _RegisterFaceScreenState extends State<RegisterFaceScreen> {
  final ImagePicker _picker = ImagePicker();
  List<XFile>? _selectedImages = [];
  bool _isUploading = false;  // 업로드 상태를 추적하기 위한 플래그

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _showWelcomeDialog(); // 화면이 처음 로드될 때 팝업 호출
    });
  }

  // 팝업을 띄우는 함수
  void _showWelcomeDialog() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('안내'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Image.asset('assets/images/face_register_ex.jpg'), // 예시 이미지
              SizedBox(height: 10),
              Text('정확한 얼굴 인식을 위해 상, 하, 좌, 우, 정면에서 촬영한 총 20장의 사진을 등록해주세요.'),
            ],
          ),
          actions: <Widget>[
            TextButton(
              child: Text('확인'),
              onPressed: () {
                Navigator.of(context).pop(); // 팝업 닫기
              },
            ),
          ],
        );
      },
    );
  }

  Future<void> _pickImages() async {
    final pickedFiles = await _picker.pickMultiImage(
      maxWidth: 800,
      maxHeight: 800,
      imageQuality: 80,
    );

    if (pickedFiles != null && pickedFiles.length <= 20) {
      setState(() {
        _selectedImages = pickedFiles;
      });
    } else if (pickedFiles != null && pickedFiles.length > 20) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('You can select up to 20 images.')),
      );
    }
  }

  // SharedPreferences에서 이메일을 가져오는 함수
  Future<String?> _getUserEmail() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    return prefs.getString('user_email');  // 로그인할 때 저장된 이메일을 가져옴
  }

  Future<void> _uploadImages() async {
    if (_selectedImages == null || _selectedImages!.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Please select some images first.')),
      );
      return;
    }

    // SharedPreferences에서 이메일 가져오기
    String? email = await _getUserEmail();

    if (email == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('User email not found. Please log in again.')),
      );
      return;
    }

    setState(() {
      _isUploading = true;  // 업로드 시작
    });

    var request = http.MultipartRequest(
      'POST',
      Uri.parse('${ApiConstants.baseUrl}/api/users/register_face/'), // Django 백엔드 API URL
    );

    // 선택한 이미지 파일들을 업로드
    for (var image in _selectedImages!) {
      var mimeType = lookupMimeType(image.path);
      var mediaType = mimeType != null ? MediaType.parse(mimeType) : null;
      request.files.add(await http.MultipartFile.fromPath(
        'faces', // 이 키는 Django에서 파일을 받을 때 사용해야 하는 키입니다.
        image.path,
        contentType: mediaType,
      ));
    }

    // 이메일도 함께 전송
    request.fields['email'] = email;

    var response = await request.send();

    setState(() {
      _isUploading = false;  // 업로드 완료
    });

    if (response.statusCode == 200) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Face registration successful.')),
      );
      // 업로드 성공 후 홈 화면으로 이동
      Navigator.of(context).pop();
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Face registration failed.')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Register Face'),
        backgroundColor: Colors.white,
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            ElevatedButton(
              onPressed: _pickImages,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.pink,
                padding: EdgeInsets.symmetric(horizontal: 40, vertical: 15),
              ),
              child: Text('Select Images', style: TextStyle(fontSize: 18)),
            ),
            SizedBox(height: 20),
            _selectedImages != null && _selectedImages!.isNotEmpty
                ? Column(
              children: [
                Text('${_selectedImages!.length} images selected'),
                SizedBox(height: 20),
                // 이미지 미리보기 추가
                Container(
                  height: 100,
                  child: ListView.builder(
                    scrollDirection: Axis.horizontal,
                    itemCount: _selectedImages!.length,
                    itemBuilder: (context, index) {
                      return Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: Image.file(
                          File(_selectedImages![index].path),
                          width: 100,
                          height: 100,
                          fit: BoxFit.cover,
                        ),
                      );
                    },
                  ),
                ),
              ],
            )
                : Text('No images selected'),
            SizedBox(height: 20),
            _isUploading
                ? CircularProgressIndicator() // 업로드 중일 때 로딩 표시
                : ElevatedButton(
              onPressed: _uploadImages,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.pink,
                padding: EdgeInsets.symmetric(horizontal: 40, vertical: 15),
              ),
              child: Text('Upload Images', style: TextStyle(fontSize: 18)),
            ),
          ],
        ),
      ),
    );
  }
}
