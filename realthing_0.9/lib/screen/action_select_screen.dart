//action_select_screen.dart
import 'package:flutter/material.dart';
import 'upload_video_screen.dart';
import 'stream_category_screen.dart';

class ActionSelectionScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Select Action'),
        backgroundColor: Colors.white,
        centerTitle: true,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            ElevatedButton(
              onPressed: () {
                // NewStreamingModal 모달을 하단에 띄움
                showModalBottomSheet(
                  context: context,
                  builder: (context) => NewStreamingModal(),
                );
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.pink,
                padding: EdgeInsets.symmetric(horizontal: 40, vertical: 15),
              ),
              child: Text('Start Live Stream', style: TextStyle(fontSize: 18)),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => UploadVideoScreen(
                      onVideoUploaded: (video) {
                        // 업로드된 비디오를 처리하는 로직
                        Navigator.pop(context, video);
                      },
                    ),
                  ),
                );
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.pink,
                padding: EdgeInsets.symmetric(horizontal: 40, vertical: 15),
              ),
              child: Text('Upload Video', style: TextStyle(fontSize: 18)),
            ),
          ],
        ),
      ),
    );
  }
}

