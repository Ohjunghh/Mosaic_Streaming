import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:video_thumbnail/video_thumbnail.dart';
import 'dart:typed_data';
import 'package:video_player/video_player.dart';
import 'stream_category_screen.dart';
import 'rtmp_view_screen.dart';
import '../constants/ip.dart';

class LiveStreamScreen extends StatefulWidget {
  @override
  _LiveStreamScreenState createState() => _LiveStreamScreenState();
}

class _LiveStreamScreenState extends State<LiveStreamScreen> {
  List<String> _liveStreams = [];
  String serverIp = '172.20.66.178';  // 서버 IP 설정
  bool _isLoading = false;
  List<String> _categories = ['All', 'Music', 'Games', 'Sports', 'Daily'];
  String _selectedCategory = 'All';

  @override
  void initState() {
    super.initState();
    _loadLiveStreams(); // 라이브 스트리밍 불러오기
  }

  // 새로고침 기능 (아래로 스크롤 시)
  Future<void> _refreshStreams() async {
    await _loadLiveStreams();  // 스트림 목록 다시 불러오기
  }

  // 할당된 ID로부터 스트림 URL 생성 및 불러오기
  Future<void> _loadLiveStreams() async {
    setState(() {
      _isLoading = true;
    });

    try {
      final response = await http.get(Uri.parse('${ApiConstants.baseUrl}/api/webcam/get_id/')); // 현재 할당된 id 가져오기
      if (response.statusCode == 200) {
        final List<dynamic> idData = jsonDecode(response.body)['ids']; // id 리스트 받아오기
        setState(() {
          _liveStreams = idData.map<String>((id) => '${ApiConstants.baseUrl}/live-out/$id.m3u8').toList();  // URL 생성
        });
      } else {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed to load live streams')));
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('An error occurred while loading live streams')));
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  // 카테고리별로 라이브 스트리밍 필터링
  List<String> get _filteredLiveStreams {
    if (_selectedCategory == 'All') {
      return _liveStreams;
    } else {
      return _liveStreams.where((stream) => _getCategory(stream) == _selectedCategory).toList();
    }
  }

  // 스트리밍 URL로부터 카테고리 추출 (예시로 URL에 카테고리가 포함되어 있다고 가정)
  String _getCategory(String streamPath) {
    if (streamPath.contains('Music')) return 'Music';
    if (streamPath.contains('Games')) return 'Games';
    if (streamPath.contains('Sports')) return 'Sports';
    if (streamPath.contains('Daily')) return 'Daily';
    return 'All';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Live Streams'),
        backgroundColor: Colors.white,
        centerTitle: true,
      ),
      body: RefreshIndicator(  // 새로고침을 위해 RefreshIndicator 추가
        onRefresh: _refreshStreams,
        child: Column(
          children: [
            _buildCategoryBar(),
            Expanded(
              child: _isLoading  // 로딩 중일 때 로딩 표시
                  ? Center(child: CircularProgressIndicator())
                  : _filteredLiveStreams.isNotEmpty
                  ? ListView.builder(
                itemCount: _filteredLiveStreams.length,
                itemBuilder: (context, index) {
                  String streamUrl = _filteredLiveStreams[index];
                  return ListTile(
                    title: Text('Stream ${index + 1}'),
                    subtitle: Text(streamUrl),
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => RTMPStreamApp(streamUrl: streamUrl), // 스트림 URL 전달
                        ),
                      );
                    },
                  );
                },
              )
                  : Center(child: Text('No live streams available.')),  // 스트림이 없을 때
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          // NewStreamingModal로 이동
          showModalBottomSheet(
            context: context,
            builder: (context) => NewStreamingModal(),
          );
        },
        child: Icon(Icons.play_arrow),
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

// 비디오 플레이어 화면
class VideoPlayerScreen extends StatefulWidget {
  final String videoPath;

  VideoPlayerScreen({required this.videoPath});

  @override
  _VideoPlayerScreenState createState() => _VideoPlayerScreenState();
}

class _VideoPlayerScreenState extends State<VideoPlayerScreen> {
  late VideoPlayerController _controller;

  @override
  void initState() {
    super.initState();
    _controller = VideoPlayerController.networkUrl(Uri.parse(widget.videoPath))
      ..initialize().then((_) {
        setState(() {}); // 비디오가 준비되면 UI 업데이트
        _controller.play();
      });
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
        title: Text('Live Stream'),
        backgroundColor: Colors.white,
        centerTitle: true,
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: Colors.pink),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: Center(
        child: _controller.value.isInitialized
            ? AspectRatio(
          aspectRatio: _controller.value.aspectRatio,
          child: VideoPlayer(_controller),
        )
            : CircularProgressIndicator(),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          setState(() {
            if (_controller.value.isPlaying) {
              _controller.pause();
            } else {
              _controller.play();
            }
          });
        },
        child: Icon(
          _controller.value.isPlaying ? Icons.pause : Icons.play_arrow,
        ),
      ),
    );
  }
}
