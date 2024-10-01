import 'package:flutter/material.dart';
import 'package:flutter_vlc_player/flutter_vlc_player.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:gallery_saver/gallery_saver.dart'; // 갤러리에 저장하기 위한 패키지

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: UrlAndStreamKeyInputPage(),
    );
  }
}

class UrlAndStreamKeyInputPage extends StatefulWidget {
  @override
  _UrlAndStreamKeyInputPageState createState() => _UrlAndStreamKeyInputPageState();
}

class _UrlAndStreamKeyInputPageState extends State<UrlAndStreamKeyInputPage> {
  final TextEditingController _urlController = TextEditingController();
  final TextEditingController _streamKeyController = TextEditingController();
  String? _rtmpUrlWithKey;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Enter RTMP URL & Stream Key'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            TextField(
              controller: _urlController,
              decoration: InputDecoration(
                labelText: 'RTMP URL',
                border: OutlineInputBorder(),
              ),
            ),
            SizedBox(height: 16),
            TextField(
              controller: _streamKeyController,
              decoration: InputDecoration(
                labelText: 'Stream Key',
                border: OutlineInputBorder(),
              ),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                String? url = _urlController.text;
                String? streamKey = _streamKeyController.text;

                // RTMP URL과 스트림 키가 입력되었는지 확인
                if (url.isNotEmpty && streamKey.isNotEmpty) {
                  _rtmpUrlWithKey = '$url/$streamKey';

                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => VlcPlayerPage(rtmpUrl: _rtmpUrlWithKey!),
                    ),
                  );
                } else {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('Please enter both RTMP URL and Stream Key')),
                  );
                }
              },
              child: Text('Start Streaming'),
            ),
          ],
        ),
      ),
    );
  }
}

class VlcPlayerPage extends StatefulWidget {
  final String rtmpUrl;

  VlcPlayerPage({required this.rtmpUrl});

  @override
  _VlcPlayerPageState createState() => _VlcPlayerPageState();
}

class _VlcPlayerPageState extends State<VlcPlayerPage> {
  late VlcPlayerController _vlcPlayerController;

  @override
  void initState() {
    super.initState();

    // VLC 플레이어 컨트롤러를 초기화합니다.
    _vlcPlayerController = VlcPlayerController.network(
      widget.rtmpUrl,
      autoPlay: true,
    );
  }

  @override
  void dispose() {
    _vlcPlayerController.dispose();
    super.dispose();
  }

  // 비디오를 갤러리에 저장하는 함수
  Future<void> _downloadVideo() async {
    try {
      // 비디오 다운로드
      var response = await http.get(Uri.parse(widget.rtmpUrl));
      if (response.statusCode == 200) {
        // 디렉토리 가져오기
        Directory? directory = await getExternalStorageDirectory();
        if (directory != null) {
          String videoPath = '${directory.path}/downloaded_video.mp4';
          File videoFile = File(videoPath);

          // 비디오 파일을 로컬에 저장
          await videoFile.writeAsBytes(response.bodyBytes);

          // 갤러리에 저장
          await GallerySaver.saveVideo(videoFile.path);
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('비디오가 갤러리에 저장되었습니다.')),
          );
        }
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('비디오 다운로드에 실패했습니다.')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('다운로드 중 오류가 발생했습니다: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('VLC Player for RTMP Stream'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            // VLC Player Widget
            AspectRatio(
              aspectRatio: 16 / 9,
              child: VlcPlayer(
                controller: _vlcPlayerController,
                aspectRatio: 16 / 9,
                placeholder: Center(child: CircularProgressIndicator()),
              ),
            ),
            SizedBox(height: 20),
          ],
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
      floatingActionButton: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          FloatingActionButton(
            heroTag: "play_pause",
            onPressed: () {
              setState(() {
                if (_vlcPlayerController.value.isPlaying) {
                  _vlcPlayerController.pause();
                } else {
                  _vlcPlayerController.play();
                }
              });
            },
            child: Icon(
              _vlcPlayerController.value.isPlaying ? Icons.pause : Icons.play_arrow,
            ),
          ),
          FloatingActionButton(
            heroTag: "download",
            onPressed: _downloadVideo, // 다운로드 함수 호출
            child: Icon(Icons.file_download),
            backgroundColor: Colors.blue,
          ),
        ],
      ),
    );
  }
}
