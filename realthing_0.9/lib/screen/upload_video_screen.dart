import 'dart:io';
import 'dart:convert'; // jsonDecode 함수를 사용하기 위한 import
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:mime/mime.dart';
import 'package:http_parser/http_parser.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:path_provider/path_provider.dart'; // path_provider 사용 추가
import 'package:flutter/services.dart';
import '../model/video_model.dart';
import '../constants/ip.dart';

class UploadVideoScreen extends StatefulWidget {
  final Function(Video) onVideoUploaded;

  UploadVideoScreen({required this.onVideoUploaded});

  @override
  _UploadVideoScreenState createState() => _UploadVideoScreenState();
}

class _UploadVideoScreenState extends State<UploadVideoScreen> {
  final ImagePicker _picker = ImagePicker();
  List<File> _videoFiles = [];
  final TextEditingController _videoNameController = TextEditingController();
  String _selectedCategory = 'Mukbang';
  final List<String> _categories = ['Mukbang', 'Games', 'Sports', 'Daily'];

  bool _isUploading = false; // 업로드 상태를 추적하기 위한 플래그

  // 모자이크 및 아바타 처리 옵션
  bool _mosaicLicensePlate = false;
  bool _mosaicInvoice = false;
  bool _mosaicIDCard = false;
  bool _mosaicLicenseCard = false;
  bool _mosaicKnife = false;
  bool _mosaicFace = false;
  bool _avatarProcessing = false;


  // SharedPreferences에서 이메일을 가져오는 함수
  Future<String?> _getUserEmail() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    return prefs.getString('user_email');
  }

  // 비디오 선택 함수
  Future<void> _pickVideo() async {
    final pickedFile = await _picker.pickVideo(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _videoFiles.add(File(pickedFile.path));
      });
    }
  }

  // 비디오 업로드 함수
  Future<void> _uploadVideos() async {
    if (_videoFiles.isEmpty || _videoNameController.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Please select videos and enter a name.')),
      );
      return;
    }

    String? email = await _getUserEmail();
    if (email == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('User email not found. Please log in again.')),
      );
      return;
    }

    setState(() {
      _isUploading = true; // 업로드 시작
    });

    for (var videoFile in _videoFiles) {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('${ApiConstants.baseUrl}/api/users/video_upload/'),
      );

      var mimeType = lookupMimeType(videoFile.path);
      var mediaType = mimeType != null ? MediaType.parse(mimeType) : null;

      request.files.add(await http.MultipartFile.fromPath(
        'video',
        videoFile.path,
        contentType: mediaType,
      ));

      request.fields['name'] = _videoNameController.text;
      request.fields['category'] = _selectedCategory;
      request.fields['email'] = email;
      request.fields['mosaic_license_plate'] = _mosaicLicensePlate.toString();
      request.fields['mosaic_invoice'] = _mosaicInvoice.toString();
      request.fields['mosaic_id_card'] = _mosaicIDCard.toString();
      request.fields['mosaic_license_card'] = _mosaicLicenseCard.toString();
      request.fields['mosaic_knife'] = _mosaicKnife.toString();
      request.fields['mosaic_face'] = _mosaicFace.toString();
      request.fields['avatar_processing'] = _avatarProcessing.toString();

      var response = await request.send();

      if (response.statusCode == 200) {
        var responseData = await http.Response.fromStream(response);
        var videoUrl = jsonDecode(responseData.body)['video_url'];

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Video uploaded successfully.')),
        );

        final video = Video(
          path: videoFile.path,
          category: _selectedCategory,
        );

        widget.onVideoUploaded(video);

        // 업로드한 비디오 다운로드 및 갤러리에 저장
        await _saveVideoToGallery(videoUrl, videoFile.path.split('/').last);

        Future.delayed(Duration(seconds: 2), () {
          Navigator.of(context).pop(); // 홈으로 이동
        });
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Video upload failed.')),
        );
      }
    }

    setState(() {
      _isUploading = false; // 업로드 완료
    });
  }

  // 서버에서 처리된 비디오 다운로드 후 갤러리에 저장하는 함수
  Future<void> _saveVideoToGallery(String videoUrl, String fileName) async {
    try {
      // 비디오를 서버에서 다운로드
      var response = await http.get(Uri.parse(videoUrl));
      if (response.statusCode == 200) {
        // 휴대폰의 외부 저장소 디렉토리 경로 가져오기
        Directory? directory = await getExternalStorageDirectory();
        if (directory != null) {
          String videoPath = '${directory.path}/$fileName';
          File videoFile = File(videoPath);

          // 다운로드한 비디오를 저장
          await videoFile.writeAsBytes(response.bodyBytes);

          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Video saved to gallery: $videoPath')),
          );
        } else {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Failed to get storage directory')),
          );
        }
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to download video')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error saving video: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Upload Videos'),
        backgroundColor: Colors.white,
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              ElevatedButton(
                onPressed: _pickVideo,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.pink,
                  padding: EdgeInsets.symmetric(horizontal: 40, vertical: 15),
                ),
                child: Text('Select Videos', style: TextStyle(fontSize: 18)),
              ),
              SizedBox(height: 20),
              _videoFiles.isNotEmpty
                  ? Text('${_videoFiles.length} videos selected')
                  : Text('No videos selected'),
              SizedBox(height: 20),
              TextField(
                controller: _videoNameController,
                decoration: InputDecoration(labelText: 'Video Name'),
              ),
              SizedBox(height: 20),
              DropdownButton<String>(
                value: _selectedCategory,
                onChanged: (String? newValue) {
                  setState(() {
                    _selectedCategory = newValue!;
                  });
                },
                items: _categories.map<DropdownMenuItem<String>>((String value) {
                  return DropdownMenuItem<String>(
                    value: value,
                    child: Text(value),
                  );
                }).toList(),
              ),
              SizedBox(height: 20),

              // 모자이크 및 아바타 처리 옵션
              Text('Mosaic Options:', style: TextStyle(fontSize: 16)),
              CheckboxListTile(
                title: Text('Mosaic License Plate'),
                value: _mosaicLicensePlate,
                onChanged: (bool? value) {
                  setState(() {
                    _mosaicLicensePlate = value!;
                  });
                },
              ),
              CheckboxListTile(
                title: Text('Mosaic Invoice'),
                value: _mosaicInvoice,
                onChanged: (bool? value) {
                  setState(() {
                    _mosaicInvoice = value!;
                  });
                },
              ),
              CheckboxListTile(
                title: Text('Mosaic ID Card'),
                value: _mosaicIDCard,
                onChanged: (bool? value) {
                  setState(() {
                    _mosaicIDCard = value!;
                  });
                },
              ),
              CheckboxListTile(
                title: Text('Mosaic License Card'),
                value: _mosaicLicenseCard,
                onChanged: (bool? value) {
                  setState(() {
                    _mosaicLicenseCard = value!;
                  });
                },
              ),
              CheckboxListTile(
                title: Text('Mosaic Knife'),
                value: _mosaicKnife,
                onChanged: (bool? value) {
                  setState(() {
                    _mosaicKnife = value!;
                  });
                },
              ),
              CheckboxListTile(
                title: Text('Mosaic Face'),
                value: _mosaicFace,
                onChanged: (bool? value) {
                  setState(() {
                    _mosaicFace = value!;
                  });
                },
              ),
              CheckboxListTile(
                title: Text('Avatar Processing'),
                value: _avatarProcessing,
                onChanged: (bool? value) {
                  setState(() {
                    _avatarProcessing = value!;
                  });
                },
              ),
              SizedBox(height: 20),
              _isUploading
                  ? CircularProgressIndicator() // 업로드 중일 때 로딩 표시
                  : ElevatedButton(
                onPressed: _uploadVideos,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.pink,
                  padding: EdgeInsets.symmetric(horizontal: 40, vertical: 15),
                ),
                child: Text('Upload Videos', style: TextStyle(fontSize: 18)),
              ),
            ],
          ),
        ),
      ),
    );
  }
}