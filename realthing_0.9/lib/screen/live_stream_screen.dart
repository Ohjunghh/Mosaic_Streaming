import 'package:http/http.dart' as http;
import 'package:apivideo_live_stream/apivideo_live_stream.dart';
import 'package:realthing/constants/setting_screen.dart';
import 'package:realthing/types/params.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:wakelock/wakelock.dart';
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../constants/constants.dart';
import '../constants/ip.dart';

MaterialColor apiVideoOrange = const MaterialColor(0xFFFA5B30, const {
  50: const Color(0xFFFBDDD4),
  100: const Color(0xFFFFD6CB),
  200: const Color(0xFFFFD1C5),
  300: const Color(0xFFFFB39E),
  400: const Color(0xFFFA5B30),
  500: const Color(0xFFF8572A),
  600: const Color(0xFFF64819),
  700: const Color(0xFFEE4316),
  800: const Color(0xFFEC3809),
  900: const Color(0xFFE53101),
});

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: LiveViewPage(),
      theme: ThemeData(
        primarySwatch: apiVideoOrange,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
    );
  }
}

class LiveViewPage extends StatefulWidget {
  const LiveViewPage({Key? key}) : super(key: key);

  @override
  _LiveViewPageState createState() => _LiveViewPageState();
}

class _LiveViewPageState extends State<LiveViewPage> with WidgetsBindingObserver {
  final ButtonStyle buttonStyle = ElevatedButton.styleFrom(textStyle: const TextStyle(fontSize: 20));
  Params config = Params();
  late final ApiVideoLiveStreamController _controller;
  bool _isStreaming = false;

  @override
  void initState() {
    WidgetsBinding.instance.addObserver(this);

    _controller = createLiveStreamController();

    _controller.initialize().catchError((e) {
      showInSnackBar(e.toString());
    });
    super.initState();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (!_controller.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      _controller.stop();
    } else if (state == AppLifecycleState.resumed) {
      _controller.startPreview();
    }
  }

  ApiVideoLiveStreamController createLiveStreamController() {
    return ApiVideoLiveStreamController(
      initialAudioConfig: config.audio,
      initialVideoConfig: config.video,
      onConnectionSuccess: () {
        print('Connection succeeded');
      },
      onConnectionFailed: (error) {
        print('Connection failed: $error');
        _showDialog(context, 'Connection failed', '$error');
        if (mounted) {
          setIsStreaming(false);
        }
      },
      onDisconnection: () {
        showInSnackBar('Disconnected');
        if (mounted) {
          setIsStreaming(false);
        }
      },
      onError: (error) {
        _showDialog(context, 'Error', '$error');
        if (mounted) {
          setIsStreaming(false);
        }
      },
    );
  }

  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: _scaffoldKey,
      appBar: AppBar(
        title: const Text('Live Stream Example'),
        actions: <Widget>[
          PopupMenuButton<String>(
            onSelected: (choice) => _onMenuSelected(choice, context),
            itemBuilder: (BuildContext context) {
              return Constants.choices.map((String choice) {
                return PopupMenuItem<String>(
                  value: choice,
                  child: Text(choice),
                );
              }).toList();
            },
          )
        ],
      ),
      body: SafeArea(
        child: Center(
          child: Column(
            children: <Widget>[
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.all(1.0),
                  child: Center(
                    child: ApiVideoCameraPreview(controller: _controller),
                  ),
                ),
              ),
              _controlRowWidget(),
            ],
          ),
        ),
      ),
    );
  }

  void _onMenuSelected(String choice, BuildContext context) {
    if (choice == Constants.Settings) {
      _awaitResultFromSettingsFinal(context);
    }
  }

  void _awaitResultFromSettingsFinal(BuildContext context) async {
    await Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => SettingsScreen(params: config)),
    );
    _controller.setVideoConfig(config.video);
    _controller.setAudioConfig(config.audio);
  }

  Widget _controlRowWidget() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: <Widget>[
        IconButton(
          icon: const Icon(Icons.cameraswitch),
          color: apiVideoOrange,
          onPressed: onSwitchCameraButtonPressed,
        ),
        IconButton(
          icon: const Icon(Icons.mic_off),
          color: apiVideoOrange,
          onPressed: onToggleMicrophoneButtonPressed,
        ),
        IconButton(
          icon: const Icon(Icons.fiber_manual_record),
          color: Colors.red,
          onPressed: !_isStreaming ? onStartStreamingButtonPressed : null,
        ),
        IconButton(
          icon: const Icon(Icons.stop),
          color: Colors.red,
          onPressed: _isStreaming ? onStopStreamingButtonPressed : null,
        ),
      ],
    );
  }

  void showInSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }

  Future<void> switchCamera() async {
    return await _controller.switchCamera();
  }

  Future<void> toggleMicrophone() async {
    return await _controller.toggleMute();
  }

  Future<void> startStreaming() async {
    final ApiVideoLiveStreamController? controller = _controller;

    if (controller == null) {
      print('Error: create a camera controller first.');
      return;
    }

    // 스트림 키가 비어있는지 확인
    if (config.streamKey.isEmpty) {
      _showDialog(context, "Error", "Stream key is not set. Please configure the stream key in settings.");
      return;
    }

    // Django 서버에 방송 시작 알리기
    await _notifyDjangoBroadcastStarted();

    return await controller.startStreaming(
      streamKey: config.streamKey,
      url: config.rtmpUrl,
    );
  }

  Future<void> _notifyDjangoBroadcastStarted() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    String? userEmail = prefs.getString('user_email'); // 저장된 이메일 불러오기
    String streamKey = config.streamKey;  // 스트림 키 가져오기

    if (userEmail == null) {
      print('User email not found. Please login first.');
      return;
    }

    try {
      final response = await http.post(
        Uri.parse('${ApiConstants.baseUrl}/api/webcam/start_flutter_stream/'),
        headers: <String, String>{
          'Content-Type': 'application/json; charset=UTF-8',
        },
        body: jsonEncode(<String, String>{
          'email': userEmail,  // 이메일 전송
          'streamKey': streamKey,  // 스트림 키 전송
        }),
      );

      if (response.statusCode == 200) {
        // 응답에서 stream_id를 받아서 저장
        final responseData = jsonDecode(response.body);
        print(response.body);
        String streamId = responseData['stream_id'];

        // stream_id를 SharedPreferences에 저장
        await prefs.setString('stream_id', streamId);

        print('Broadcast start notification sent successfully, stream_id: $streamId');
      } else {
        print('Failed to notify Django: ${response.statusCode}');
      }
    } catch (e) {
      print('Error notifying Django: $e');
    }
  }

  Future<void> stopStreaming() async {
    final ApiVideoLiveStreamController? controller = _controller;

    if (controller == null) {
      print('Error: create a camera controller first.');
      return;
    }

    // Django 서버에 방송 종료 알리기
    await _notifyDjangoBroadcastStopped();

    return await controller.stopStreaming();
  }

  Future<void> _notifyDjangoBroadcastStopped() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    String? userEmail = prefs.getString('user_email');  // 저장된 이메일 불러오기
    String? streamId = prefs.getString('stream_id');  // 저장된 stream_id 불러오기

    if (userEmail == null || streamId == null) {
      print('User email or stream_id not found.');
      return;
    }

    try {
      final response = await http.post(
        Uri.parse('${ApiConstants.baseUrl}/api/webcam/stop_flutter_stream/'),  // 방송 종료 URL
        headers: <String, String>{
          'Content-Type': 'application/json; charset=UTF-8',
        },
        body: jsonEncode(<String, String>{
          'email': userEmail,  // 이메일 전송
          'stream_id': streamId,  // 저장된 stream_id 전송
        }),
      );

      if (response.statusCode == 200) {
        print('Broadcast stop notification sent successfully');
      } else {
        print('Failed to notify Django: ${response.statusCode}');
      }
    } catch (e) {
      print('Error notifying Django: $e');
    }
  }

  void onSwitchCameraButtonPressed() {
    switchCamera().then((_) {
      if (mounted) {
        setState(() {});
      }
    }).catchError((error) {
      _showDialog(context, "Error", "Failed to switch camera: $error");
    });
  }

  void onToggleMicrophoneButtonPressed() {
    toggleMicrophone().then((_) {
      if (mounted) {
        setState(() {});
      }
    }).catchError((error) {
      _showDialog(context, "Error", "Failed to toggle mute: $error");
    });
  }

  void onStartStreamingButtonPressed() {
    startStreaming().then((_) {
      if (mounted) {
        setIsStreaming(true);
      }
    }).catchError((error) {
      _showDialog(context, "Error", "Failed to start stream: $error");
    });
  }

  void onStopStreamingButtonPressed() {
    stopStreaming().then((_) {
      if (mounted) {
        setIsStreaming(false);
      }
    }).catchError((error) {
      _showDialog(context, "Error", "Failed to stop stream: $error");
    });
  }

  void setIsStreaming(bool isStreaming) {
    setState(() {
      if (isStreaming) {
        Wakelock.enable();
      } else {
        Wakelock.disable();
      }
      _isStreaming = isStreaming;
    });
  }

  Future<void> _showDialog(BuildContext context, String title, String description) async {
    return showDialog<void>(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text(title),
          content: SingleChildScrollView(child: Text(description)),
          actions: <Widget>[
            TextButton(
              child: const Text('Dismiss'),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }
}
