import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:video_thumbnail/video_thumbnail.dart';
import 'dart:typed_data';
import 'package:video_player/video_player.dart';
import 'stream_category_screen.dart';

class LiveStreamScreen extends StatefulWidget {
  @override
  _LiveStreamScreenState createState() => _LiveStreamScreenState();
}

class _LiveStreamScreenState extends State<LiveStreamScreen> {
  List<String> _liveStreams = [];
  List<String> _categories = ['All', 'Music', 'Games', 'Sports', 'Daily'];
  String _selectedCategory = 'All';

  @override
  void initState() {
    super.initState();
    _loadLiveStreams(); // 라이브 스트리밍 불러오기
  }

  // 라이브 스트리밍 불러오기
  Future<void> _loadLiveStreams() async {
    try {
      final response = await http.get(Uri.parse('http://localhost:8000/api/live/stream/')); // 라이브 스트리밍 엔드포인트
      if (response.statusCode == 200) {
        final List<dynamic> streamData = jsonDecode(response.body);

        setState(() {
          _liveStreams = streamData
              .where((data) => data is Map<String, dynamic> && data.containsKey('stream_url'))
              .map<String>((data) => data['stream_url'].toString())
              .toList();
        });
      } else {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed to load live streams')));
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('An error occurred while loading live streams')));
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
      body: Column(
        children: [
          _buildCategoryBar(),
          Expanded(
            child: ListView(
              children: [
                _filteredLiveStreams.isNotEmpty
                    ? _buildStreamSection('Live Streams', _filteredLiveStreams)
                    : Center(child: Text('No live streams available.')),
              ],
            ),
          ),
        ],
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

  // 스트리밍 섹션 빌드
  Widget _buildStreamSection(String title, List<String> streams) {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          ),
          SizedBox(height: 8.0),
          GridView.builder(
            shrinkWrap: true,
            physics: NeverScrollableScrollPhysics(),
            itemCount: streams.length,
            gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 2,
              crossAxisSpacing: 8.0,
              mainAxisSpacing: 8.0,
              childAspectRatio: 16 / 9,
            ),
            itemBuilder: (context, index) {
              String streamUrl = streams[index];
              return GestureDetector(
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => VideoPlayerScreen(videoPath: streamUrl), // 스트리밍 재생
                    ),
                  );
                },
                child: FutureBuilder<Uint8List?>(
                  future: _generateThumbnail(streamUrl),
                  builder: (context, snapshot) {
                    if (snapshot.connectionState == ConnectionState.done && snapshot.hasData) {
                      return Stack(
                        children: [
                          Container(
                            decoration: BoxDecoration(
                              image: DecorationImage(
                                image: MemoryImage(snapshot.data!),
                                fit: BoxFit.cover,
                              ),
                              borderRadius: BorderRadius.circular(8.0),
                            ),
                          ),
                          Positioned(
                            bottom: 8.0,
                            left: 8.0,
                            child: Container(
                              padding: EdgeInsets.symmetric(horizontal: 8.0, vertical: 4.0),
                              color: Colors.black54,
                              child: Text(
                                streamUrl.split('/').last,
                                style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                              ),
                            ),
                          ),
                        ],
                      );
                    } else {
                      return Container(
                        decoration: BoxDecoration(
                          color: Colors.grey[300],
                          borderRadius: BorderRadius.circular(8.0),
                        ),
                        child: Center(child: CircularProgressIndicator()),
                      );
                    }
                  },
                ),
              );
            },
          ),
        ],
      ),
    );
  }

  // 썸네일 생성
  Future<Uint8List?> _generateThumbnail(String streamUrl) async {
    try {
      final uint8list = await VideoThumbnail.thumbnailData(
        video: streamUrl,
        imageFormat: ImageFormat.PNG,
        maxWidth: 128,
        quality: 25,
      );
      return uint8list;
    } catch (e) {
      print("Error generating thumbnail: $e");
      return null;
    }
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
