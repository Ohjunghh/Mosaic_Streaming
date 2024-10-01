import 'package:apivideo_live_stream/apivideo_live_stream.dart';

// Resolution을 int로 변환하는 확장 함수 추가
extension ResolutionToInt on Resolution {
  int toInt() {
    return Resolution.values.indexOf(this);  // Resolution을 인덱스로 변환
  }
}

// Resolution을 문자열로 변환하는 확장 함수 (기존)
extension ResolutionExtension on Resolution {
  String toPrettyString() {
    switch (this) {
      case Resolution.RESOLUTION_240:
        return "352x240";
      case Resolution.RESOLUTION_360:
        return "640x360";
      case Resolution.RESOLUTION_480:
        return "858x480";
      case Resolution.RESOLUTION_720:
        return "1280x720";
      case Resolution.RESOLUTION_1080:
        return "1920x1080";
      default:
        return "1280x720";
    }
  }
}

// Resolution을 Map<int, String>으로 변환하는 함수
Map<int, String> getResolutionsMap() {
  Map<int, String> map = {};
  for (final res in Resolution.values) {
    map[res.toInt()] = res.toPrettyString();  // Resolution을 int로 변환하여 맵에 저장
  }
  return map;
}
