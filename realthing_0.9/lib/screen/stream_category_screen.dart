import 'package:flutter/material.dart';
import 'live_stream_screen.dart';

class NewStreamingModal extends StatefulWidget {
  const NewStreamingModal({Key? key}) : super(key: key);

  @override
  State<NewStreamingModal> createState() => _NewStreamingModalState();
}

class _NewStreamingModalState extends State<NewStreamingModal> {
  FocusNode? _titleFNode;
  int _selectedCategoryIndex = 0;
  String? _title;
  bool _isTitleValid = true;

  final List<String> _categories = ['All', 'Mukbang', 'Games', 'Sports', 'Daily'];

  @override
  void initState() {
    super.initState();
    _titleFNode = FocusNode();
  }

  @override
  void dispose() {
    _titleFNode?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        _titleFNode!.unfocus();
      },
      child: SingleChildScrollView(
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          height: MediaQuery.of(context).size.height * 0.9, // 화면 높이의 90%로 설정
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Center(
                child: Text(
                  "새로운 스트리밍 시작",
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
              ),
              const SizedBox(height: 20),
              _titleTextField(),
              const SizedBox(height: 20),
              _categorySelectSection(),
              const SizedBox(height: 30),
              _confirmBtn(context),
            ],
          ),
        ),
      ),
    );
  }

  Widget _titleTextField() {
    return TextFormField(
      focusNode: _titleFNode,
      onChanged: (value) {
        setState(() {
          _title = value;
          _isTitleValid = true;
        });
      },
      decoration: InputDecoration(
        labelText: "스트리밍 제목",
        labelStyle: TextStyle(color: _isTitleValid ? Colors.black54 : Colors.red),
        hintText: "제목을 입력해주세요",
        hintStyle: TextStyle(color: Colors.black26),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide(color: Colors.black12),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide(color: _isTitleValid ? Colors.blue : Colors.red),
        ),
      ),
    );
  }

  Widget _categorySelectSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          "카테고리 선택:",
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 10),
        Wrap(
          spacing: 10,
          children: List.generate(_categories.length, (index) {
            return ChoiceChip(
              label: Text(_categories[index]),
              selected: _selectedCategoryIndex == index,
              onSelected: (selected) {
                setState(() {
                  _selectedCategoryIndex = index;
                });
              },
              selectedColor: Colors.blue,
              backgroundColor: Colors.grey.shade200,
              labelStyle: TextStyle(
                color: _selectedCategoryIndex == index ? Colors.white : Colors.black,
              ),
            );
          }),
        ),
      ],
    );
  }

  Widget _confirmBtn(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton(
        onPressed: () {
          if (_title == null || _title!.isEmpty) {
            _checkTitleField();
          } else {
            Navigator.pushReplacement(
              context,
              MaterialPageRoute(builder: (context) => const LiveViewPage()),
            );
          }
        },
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.blue,
          padding: const EdgeInsets.symmetric(vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
          ),
        ),
        child: const Text(
          "스트리밍 시작",
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
      ),
    );
  }

  void _checkTitleField() {
    FocusScope.of(context).requestFocus(_titleFNode);
    setState(() {
      _isTitleValid = false;
    });
  }
}


