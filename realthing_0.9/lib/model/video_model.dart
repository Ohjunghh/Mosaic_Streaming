class Video {
  final String path;
  final String category;

  Video({required this.path, required this.category});

  // fromJson 메서드 추가
  factory Video.fromJson(Map<String, dynamic> json) {
    return Video(
      path: json['video_file'] as String,
      category: json['category'] as String,
    );
  }
}