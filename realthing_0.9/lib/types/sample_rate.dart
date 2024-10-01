import 'package:apivideo_live_stream/apivideo_live_stream.dart';

// SampleRate를 int로 변환하는 확장 함수
extension SampleRateToInt on SampleRate {
  int toInt() {
    return SampleRate.values.indexOf(this);  // SampleRate를 인덱스로 변환
  }
}

// SampleRate를 문자열로 변환하는 확장 함수 (기존)
extension SampleRateExtension on SampleRate {
  String toPrettyString() {
    switch (this) {
      case SampleRate.kHz_11:
        return "11 kHz";
      case SampleRate.kHz_22:
        return "22 kHz";
      case SampleRate.kHz_44_1:
        return "44.1 kHz";
      default:
        return "";
    }
  }
}

// SampleRate를 Map<int, String>으로 변환하는 함수
Map<int, String> getSampleRatesMap() {
  Map<int, String> map = {};
  for (final rate in SampleRate.values) {
    map[rate.toInt()] = rate.toPrettyString();  // SampleRate를 int로 변환하여 맵에 저장
  }
  return map;
}
