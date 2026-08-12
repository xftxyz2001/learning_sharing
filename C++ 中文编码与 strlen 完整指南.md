# C++ 中文编码与 strlen 完整指南

本文整理了 `const char* s = "中文";` 搭配 `strlen` 的结果、编码原理、修改编码方式、字符统计方案，全平台通用，可直接复制使用。

## 一、核心问题：strlen ("中文") 结果是多少？

代码：

cpp

运行

```cpp
const char* s = "中文";
```

-   `strlen` **只统计字节数，不统计中文字符个数**
-   结果由**系统默认编码**决定：
    
    表格
    
    | 操作系统 | 默认编码 | strlen 结果 | 原理  |
    | --- | --- | --- | --- |
    | Windows 简体中文 | GBK | 4   | 1 个中文 = 2 字节 |
    | Linux/macOS | UTF-8 | 6   | 1 个中文 = 3 字节 |
    

* * *

## 二、编码基础概念

1.  **`char*` 是单字节指针**：存储字符串以**字节**为单位，中文属于多字节字符
2.  **两种核心编码**：
    
    -   GBK：Windows 旧标准，中文占用 2 字节
    -   UTF-8：现代跨平台标准，中文占用 3 字节
    
3.  **`strlen` 规则**：从指针开始计数，遇到 `\0` 停止，**不计算结束符**。

* * *

## 三、如何修改字符串编码格式（最实用）

### 1\. 强制 UTF-8 编码（推荐，跨平台统一）

添加 `u8` 前缀，无视系统，固定为 UTF-8：

cpp

运行

```cpp
// 强制 UTF-8，所有平台 strlen 都=6
const char* s = u8"中文";
```

### 2\. 强制 GBK 编码（仅 Windows）

不添加前缀，Windows 默认就是 GBK：

cpp

运行

```cpp
// Windows 下为 GBK，strlen=4
const char* s = "中文";
```

* * *

## 四、统计真正的中文字符个数（=2）

`strlen` 无法统计字符数，用以下方案：

### 方案 1：宽字符 `wchar_t`（最简单）

cpp

运行

```cpp
#include <iostream>
#include <cwchar>
using namespace std;

int main() {
    const wchar_t* s = L"中文";
    // 直接统计字符数，结果=2
    wcout << L"字符个数：" << wcslen(s) << endl; 
    return 0;
}
```

### 方案 2：C++17 标准 UTF-8 字符串（最规范）

cpp

运行

```cpp
#include <string>
int main() {
    std::u8string s = u8"中文";
    return 0;
}
```

* * *

## 五、VS 编辑器编码设置（解决乱码）

### 1\. 源码文件保存为 UTF-8

文件 → 另存为 → 点击「保存」旁下拉 → **编码保存**

选择：**UTF-8 无签名**

### 2\. 控制台输出 UTF-8 不乱码

cpp

运行

```cpp
// Windows 控制台切换为 UTF-8 编码
system("chcp 65001");
```

* * *

## 六、完整测试代码

cpp

运行

```cpp
#include <iostream>
#include <cstring>
using namespace std;

int main() {
    // 1. 默认编码（Windows=GBK,Linux=UTF-8）
    const char* s1 = "中文";
    cout << "默认编码字节数：" << strlen(s1) << endl;

    // 2. 强制 UTF-8 编码（全平台=6）
    const char* s2 = u8"中文";
    cout << "UTF-8 编码字节数：" << strlen(s2) << endl;

    // 3. 宽字符统计字符数（=2）
    const wchar_t* s3 = L"中文";
    wcout << L"中文字符个数：" << wcslen(s3) << endl;

    return 0;
}
```

* * *

## 七、总结

1.  `strlen` 统计**字节数**，不是字符数
2.  跨平台统一编码：用 `u8"中文"`（UTF-8）
3.  Windows 专用 GBK：直接写 `"中文"`
4.  统计中文字符个数：用 `wchar_t` + `wcslen()`

