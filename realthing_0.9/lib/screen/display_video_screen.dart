//display_video_screen.dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:video_player/video_player.dart';

class DisplayVideoScreen extends StatefulWidget {
  final String downloadUrl;

  DisplayVideoScreen({required this.downloadUrl});

  @override
  _DisplayVideoScreenState createState() => _DisplayVideoScreenState();
}

class _DisplayVideoScreenState extends State<DisplayVideoScreen> {
  late VideoPlayerController _controller;
  bool _isVideoDownloaded = false;
  String? _videoPath;

  @override
  void initState() {
    super.initState();
    _downloadAndPlayVideo(widget.downloadUrl);
  }

  Future<void> _downloadAndPlayVideo(String url) async {
    final response = await http.get(Uri.parse(url));

    if (response.statusCode == 200) {
      final Directory tempDir = await getTemporaryDirectory();
      final String tempPath = tempDir.path;
      final File file = File('$tempPath/downloaded_video.mp4');

      await file.writeAsBytes(response.bodyBytes);

      setState(() {
        _isVideoDownloaded = true;
        _videoPath = file.path;
        _controller = VideoPlayerController.file(File(_videoPath!))
          ..initialize().then((_) {
            setState(() {});
            _controller.play();
          });
      });
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Video download failed.')),
      );
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Downloaded Video'),
        backgroundColor: Colors.white,
        centerTitle: true,
      ),
      body: Center(
        child: _isVideoDownloaded && _controller.value.isInitialized
            ? AspectRatio(
          aspectRatio: _controller.value.aspectRatio,
          child: VideoPlayer(_controller),
        )
            : CircularProgressIndicator(),
      ),
      floatingActionButton: _isVideoDownloaded
          ? FloatingActionButton(
        onPressed: () {
          setState(() {
            _controller.value.isPlaying
                ? _controller.pause()
                : _controller.play();
          });
        },
        child: Icon(
          _controller.value.isPlaying ? Icons.pause : Icons.play_arrow,
        ),
      )
          : null,
    );
  }
}
