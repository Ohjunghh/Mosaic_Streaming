// rtmp_view_screen.dart
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

void main() {
  runApp(RTMPStreamApp(streamUrl: 'http://example.com/stream.m3u8')); // 기본 URL
}

class RTMPStreamApp extends StatelessWidget {
  final String streamUrl;

  RTMPStreamApp({required this.streamUrl});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'HLS Stream Player',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: VideoStreamPage(streamUrl: streamUrl),
    );
  }
}

class VideoStreamPage extends StatefulWidget {
  final String streamUrl;

  VideoStreamPage({required this.streamUrl});

  @override
  _VideoStreamPageState createState() => _VideoStreamPageState();
}

class _VideoStreamPageState extends State<VideoStreamPage> {
  late VideoPlayerController _controller;
  bool _isPlaying = false;

  @override
  void initState() {
    super.initState();
    _controller = VideoPlayerController.networkUrl(
      Uri.parse(widget.streamUrl), // 전달받은 스트림 URL 사용
    )
      ..initialize().then((_) {
        setState(() {
          _controller.play(); // 초기화 후 자동 재생
          _isPlaying = true;
        });
      }).catchError((error) {
        print("Error loading video: $error");  // 오류 발생 시 콘솔에 출력
      });
  }

  @override
  void dispose() {
    _controller.dispose(); // 리소스 해제
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Live Stream'),
      ),
      body: Center(
        child: _controller.value.isInitialized
            ? AspectRatio(
          aspectRatio: _controller.value.aspectRatio,
          child: VideoPlayer(_controller),
        )
            : CircularProgressIndicator(), // 초기화되지 않으면 로딩 스피너 표시
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          setState(() {
            if (_controller.value.isPlaying) {
              _controller.pause(); // 재생 중이면 일시 정지
              _isPlaying = false;
            } else {
              _controller.play(); // 일시 정지 상태면 재생
              _isPlaying = true;
            }
          });
        },
        child: Icon(
          _isPlaying ? Icons.pause : Icons.play_arrow,
        ),
      ),
    );
  }
}
