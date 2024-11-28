// 마이 비디오 스크린
import 'package:flutter/material.dart';
// import 'package:video_player/video_player.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:video_thumbnail/video_thumbnail.dart';
import 'dart:typed_data';
import 'dart:io'; // File 사용 추가
import 'package:path_provider/path_provider.dart'; // path_provider 사용 추가
import 'upload_video_screen.dart'; // 업로드 화면 import
import '../model/video_model.dart'; // Video 모델 import
import 'video_player_screen.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../constants/ip.dart';

class MyVideosScreen extends StatefulWidget {
  @override
  _MyVideosScreenState createState() => _MyVideosScreenState();
}

class _MyVideosScreenState extends State<MyVideosScreen> {
  List<String> _uploadedVideos = [];  // 업로드된 비디오 목록
  List<String> _processedVideos = []; // 처리된 비디오 목록
  List<String> _categories = ['All', 'Mukbang', 'Games', 'Sports', 'Daily'];
  String _selectedCategory = 'All';

  @override
  void initState() {
    super.initState();
    _loadProcessedVideos(); // 처리된 비디오 불러오기
  }

  // SharedPreferences에서 이메일을 가져오는 함수
  Future<String?> _getUserEmail() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    return prefs.getString('user_email');
  }

  // 처리된 비디오 불러오기
  Future<void> _loadProcessedVideos() async {
    //String userEmail = 'aaa@naver.com';
    String? userEmail = await _getUserEmail(); //'aaa@naver.com';
    try {
      final response = await http.get(
          Uri.parse('${ApiConstants.baseUrl}/api/users/video_download/?email=$userEmail'));
      print('Response status: ${response.statusCode}');
      print('Response body: ${response.body}');

      if (response.statusCode == 200) {
        final List<dynamic> videoData = jsonDecode(response.body)['processed_videos']; // 응답에서 'processed_videos' 추출
        List<String> localVideoPaths = [];

        for (var video in videoData) {
          String videoUrl = video['video_url'].toString();
          String localPath = await _downloadAndSaveVideo(videoUrl, video['name'].toString());

          // 중복 체크: 이미 추가된 경로는 중복으로 추가하지 않음
          if (!_processedVideos.contains(localPath)) {
            localVideoPaths.add(localPath);
          }
        }

        setState(() {
          // 기존 처리된 비디오 목록을 유지하면서 새로운 비디오 목록 추가
          _processedVideos = List.from(_processedVideos)..addAll(localVideoPaths);
          print('Processed videos: $_processedVideos');
        });
      } else {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed to load processed videos')));
      }
    } catch (e) {
      print('Error occurred: $e');
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('An error occurred while loading processed videos')));
    }
  }


  // 서버에서 비디오 다운로드 후 로컬에 저장
  Future<String> _downloadAndSaveVideo(String videoUrl, String fileName) async {
    try {
      var response = await http.get(Uri.parse(videoUrl));
      if (response.statusCode == 200) {
        Directory? directory = await getExternalStorageDirectory();
        if (directory != null) {
          String videoPath = '${directory.path}/$fileName';
          File videoFile = File(videoPath);
          await videoFile.writeAsBytes(response.bodyBytes);
          return videoPath; // 저장된 로컬 경로 반환
        } else {
          throw Exception('Failed to get directory');
        }
      } else {
        throw Exception('Failed to download video');
      }
    } catch (e) {
      throw Exception('Error downloading video: $e');
    }
  }

  // 썸네일 생성 (로컬 비디오 파일에 대해 썸네일 생성)
  Future<Uint8List?> _generateThumbnail(String videoPath) async {
    try {
      final uint8list = await VideoThumbnail.thumbnailData(
        video: videoPath,
        imageFormat: ImageFormat.PNG,
        maxWidth: 128, // 썸네일의 최대 너비
        quality: 25,
        timeMs: 1000,
      );
      return uint8list;
    } catch (e) {
      print("Error generating thumbnail: $e");
      return null;
    }
  }

  // 동영상 업로드 후 콜백 함수 (업로드된 비디오 목록에 추가)
  void _onVideoUploaded(Video video) {
    setState(() {
      // 기존 업로드된 비디오 목록에 새 비디오 추가
      _uploadedVideos = List.from(_uploadedVideos)..add(video.path);
    });

    // 처리된 비디오도 새로 고침
    _loadProcessedVideos();
  }

  // 업로드된 비디오 섹션 (썸네일 포함)
  Widget _buildUploadedVideoList(List<String> uploadedVideos) {
    return ListView.builder(
      shrinkWrap: true,
      physics: NeverScrollableScrollPhysics(),
      itemCount: uploadedVideos.length,
      itemBuilder: (context, index) {
        String videoPath = uploadedVideos[index];

        return ListTile(
          leading: FutureBuilder<Uint8List?>(
            future: _generateThumbnail(videoPath), // 썸네일 생성 함수 호출
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.done && snapshot.hasData) {
                return Container(
                  width: 100, // 썸네일의 너비
                  height: 60, // 썸네일의 높이
                  decoration: BoxDecoration(
                    image: DecorationImage(
                      image: MemoryImage(snapshot.data!), // 썸네일 이미지 표시
                      fit: BoxFit.cover,
                    ),
                    borderRadius: BorderRadius.circular(8.0),
                  ),
                );
              } else {
                return Container(
                  width: 100, // 썸네일의 너비
                  height: 60, // 썸네일의 높이
                  color: Colors.grey[300], // 썸네일을 로딩 중일 때 보여줄 색상
                  child: Center(child: CircularProgressIndicator()), // 로딩 중 스피너
                );
              }
            },
          ),
          title: Text('Uploaded Video: ${videoPath.split('/').last}'), // 업로드된 비디오 파일명 표시
          onTap: () {
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) => VideoPlayerScreen(videoPath: videoPath),
              ),
            );
          },
        );
      },
    );
  }

  // 처리된 비디오 섹션 (썸네일 포함, 로컬 경로 사용)
  Widget _buildProcessedVideoList(List<String> processedVideos) {
    return ListView.builder(
      shrinkWrap: true,
      physics: NeverScrollableScrollPhysics(),
      itemCount: processedVideos.length,
      itemBuilder: (context, index) {
        String videoPath = processedVideos[index];

        return ListTile(
          leading: FutureBuilder<Uint8List?>(
            future: _generateThumbnail(videoPath), // 처리된 비디오 썸네일 생성
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.done && snapshot.hasData) {
                return Container(
                  width: 100,
                  height: 60,
                  decoration: BoxDecoration(
                    image: DecorationImage(
                      image: MemoryImage(snapshot.data!),
                      fit: BoxFit.cover,
                    ),
                    borderRadius: BorderRadius.circular(8.0),
                  ),
                );
              } else {
                return Container(
                  width: 100,
                  height: 60,
                  color: Colors.grey[300],
                  child: Center(child: CircularProgressIndicator()),
                );
              }
            },
          ),
          title: Text('Processed Video: ${videoPath.split('/').last}'), // 처리된 비디오 파일명 표시
          onTap: () {
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) => VideoPlayerScreen(videoPath: videoPath),
              ),
            );
          },
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: RefreshIndicator(
        onRefresh: _loadProcessedVideos, // 새로고침 시 처리된 비디오 목록 갱신
        child: SingleChildScrollView(
          child: Column(
            children: [
              _buildCategoryBar(), // 카테고리 선택 바
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // 처리된 비디오 섹션 (썸네일 포함, 업로드된 비디오와 동일한 형식)
                    _processedVideos.isNotEmpty
                        ? _buildProcessedVideoList(_processedVideos)
                        : Center(child: Text('처리된 비디오가 없습니다.')),

                    // 구분선 추가
                    Divider(thickness: 2, color: Colors.grey[400]),

                    // 업로드된 비디오 목록 섹션 (썸네일 포함)
                    _uploadedVideos.isNotEmpty
                        ? _buildUploadedVideoList(_uploadedVideos)
                        : Center(child: Text('업로드된 비디오가 없습니다.')),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => UploadVideoScreen(
                onVideoUploaded: _onVideoUploaded, // 업로드 후 콜백 함수 전달
              ),
            ),
          );
        },
        child: Icon(Icons.add),
        backgroundColor: Colors.pink,
      ),
    );
  }

  // 카테고리 선택 바
  Widget _buildCategoryBar() {
    return Container(
      height: 50,
      padding: EdgeInsets.symmetric(vertical: 8.0),
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        itemCount: _categories.length,
        itemBuilder: (context, index) {
          String category = _categories[index];
          bool isSelected = _selectedCategory == category;
          return GestureDetector(
            onTap: () {
              setState(() {
                _selectedCategory = category;
              });
            },
            child: Container(
              margin: EdgeInsets.symmetric(horizontal: 8.0),
              padding: EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
              decoration: BoxDecoration(
                color: isSelected ? Colors.pink : Colors.grey[300],
                borderRadius: BorderRadius.circular(20.0),
              ),
              child: Center(
                child: Text(
                  category,
                  style: TextStyle(
                    color: isSelected ? Colors.white : Colors.black,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}
