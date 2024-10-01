import 'package:apivideo_live_stream/apivideo_live_stream.dart';

// Channel을 int로 변환하는 확장 함수
extension ChannelToInt on Channel {
  int toInt() {
    return Channel.values.indexOf(this);  // Channel을 인덱스로 변환
  }
}

// Channel을 문자열로 변환하는 확장 함수 (기존)
extension ChannelExtension on Channel {
  String toPrettyString() {
    switch (this) {
      case Channel.mono:
        return "mono";
      case Channel.stereo:
        return "stereo";
      default:
        return "stereo";
    }
  }
}

// Channel을 Map<int, String>으로 변환하는 함수
Map<int, String> getChannelsMap() {
  Map<int, String> map = {};
  for (final channel in Channel.values) {
    map[channel.toInt()] = channel.toPrettyString();  // Channel을 int로 변환하여 맵에 저장
  }
  return map;
}
