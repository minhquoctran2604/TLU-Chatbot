![Image](/content/drive/MyDrive/MD/CNWeeb/8-CNWeb-PHP-Laravel_artifacts/image_000000_7077877430444fb4b190f710d6a0f2603f4e160e9d6e4974156e13cfdf22bc6d.png)

![Image](/content/drive/MyDrive/MD/CNWeeb/8-CNWeb-PHP-Laravel_artifacts/image_000001_60cd87004c4cc6272817c79b4fb6008f7031d4013ac4f7ccb9284841474b144f.png)

Giảng viên: Kiều Tuấn Dũng , Nguyễn Tu Trung BM HTTT, Khoa CNTT, Trường ĐH Thủy Lợi

Hà Nội, 2024

## Nôi dung

- ❖ Laravel là gì
- ❖ Khái niệm Composer
- ❖ Cài đặt Laravel sử dụng Composer
- ❖ Cấu trúc thư mục Laravel
- ❖ Luồng xử lý trong Laravel
- ❖ File môi trường .env
- ❖ Routing
- ❖ Middleware
- ❖ Namespace
- ❖ Controller
- ❖ Request
- ❖ Cookie
- ❖ Session
- ❖ Views -Blade Template
- ❖ Views -Layout
- ❖ Redirect
- ❖ Làm việc với Database
- ❖ Validate Form

![Image](/content/drive/MyDrive/MD/CNWeeb/8-CNWeb-PHP-Laravel_artifacts/image_000002_e46593a0174baf2f2fe5051b93cf0890e4cac313bcfa2f25106d9a39b1b8c9b5.png)

- ❖ Là framework mã nguồn mở, hoạt động theo mô hình MVC
- ❖ Được sử dụng phổ biến nhất trong các framework hiện tại
- ❖ Trang chủ: https://laravel.com
- ❖ Document: https://laravel.com/docs/25.x

## Khái niệm Composer

- ❖ Tải về và cài đặt tại https://getcomposer.org/
- ❖ Công cụ quản lý việc cài đặt, update tự động các thư viện code của bên thứ ba
- ❖ Là công cụ không thể thiếu đối với lập trình
- ❖ Laravel sẽ được cài đặt thông qua Composer

## Cài đặt Laravel sử dụng Composer

- ❖ Cách 1: chạy lần lượt 2 lệnh sau:
- ❖ composer global require laravel/installer
- ❖ laravel new &lt;tên -thư -mục -project&gt;
- ❖ Cách 2:
- ❖ composer create -project --prefer-dist laravel/laravel &lt;tên -thư -mục -project&gt;

## Cấu trúc thư mục Laravel

TH1

![Image](/content/drive/MyDrive/MD/CNWeeb/8-CNWeb-PHP-Laravel_artifacts/image_000003_5fca68c1c897b3b3c3bfe7f2e5a81cf8e0838f9e80d0cff5ac7a57e78f87055b.png)

## Luồng xử lý trong Laravel

![Image](/content/drive/MyDrive/MD/CNWeeb/8-CNWeb-PHP-Laravel_artifacts/image_000004_f5e45d22f3e22f23e95b639e2f81b0b1d3a2b27e21e6e23987d4366dab82d9a8.png)

## File môi trường .env

- ❖ File chứa các thông tin quan trọng liên quan đến cấu hình hệ thống và database
- ❖ Khi thay đổi thông tin cấu hình DB sẽ thay đổi tại file .env này mà không cần vào trong thư mục config/database.php
- ❖ Với CSDL MySql trên local sẽ thay đổi các thông số sau

DB\_CONNECTION=mysql DB\_HOST=127.0.0.1 DB\_PORT=3306 DB\_DATABASE=tên-CSDL DB\_USERNAME=root DB\_PASSWORD=

## Routing

- ❖ Trong Laravel, mọi url đều phải thông qua cơ chế Routing
- ❖ Các route nằm tại routes/web.php
- ❖ Route có thể có tham số hoặc không
- ❖ Ví dụ:

```
Route::get('demo-page', function () { return view('demo-page'); }); Route::get('param2/{name}/{age}', function ($name, $age) { echo "Name = $name <br />"; echo "Age = $age <br />"; }); Route::get('/home', 'HomeController@index');
```

## Middleware

- ❖ Đóng vai trò như 1 firewall, lọc request gửi tới và response trả về
- ❖ Lệnh tạo với artisan: php artisan make:middleware &lt;tênmiddleware&gt;
- ❖ Sau khi tạo, đăng ký middleware tại app/Http/Kernel.php
- ❖ Có thể sử dụng middleware khi tạo route hoặc tại hàm khởi tạo của controller
- ❖ Ví dụ đặt tại route của middleware FirstMiddleware

```
Route::get('demo-middleware/{id}', function ($id) { echo "Đã pass qua middleware FirstMiddleware"; })->middleware('first');
```

## Namespace

- ❖ Được sử dụng như 1 định danh duy nhất cho 1 file, giải quyết trường hợp import các file trùng tên trong cùng 1 ứng dụng
- ❖ Laravel sử dụng từ khóa use để import class dựa theo namespace, thay vì phải import các file sử dụng các hàm include/require trong MVC thuần
- ❖ VD: import class Request có namespace là Illuminate\Http

use Illuminate\Http\Request;

## Controller

- ❖ Ví trị tại app/http/Controllers
- ❖ Tạo với artisan:
- ❖ php artisan make:controller &lt;tên-controller&gt;
- ❖ Tạo controller với các phương thức CRUD chuẩn của Laravel
- ❖ php artisan make:controller &lt;tên-controller&gt; --resource

## Request

- ❖ Chứa toàn bộ các thông tin của request gửi đến
- ❖ Sử dụng thông qua đối tượng của lớp Request hoặc phương thức request()
- ❖ Lấy thông tin liên quan đến url
- ❖ request()-&gt;path(); //lấy đường dẫn
- ❖ $request-&gt;path(); //lấy đường dẫn
- ❖ $request-&gt;url(); //lấy url, bỏ qua tham số truy vấn nếu có
- ❖ Lấy thông tin từ form, có các cách lấy sau
- ❖ $request-&gt;input('username');
- ❖ $request-&gt;username;
- ❖ Hai cách trên đều dùng để lấy giá trị input form có name là username

## Session

- ❖ Nên sử dụng các hàm thao tác với session mà Laravel cung cấp, thay vì sử dụng biến $\_SESSION của PHP
- ❖ Set giá trị: session()-&gt;put(key, value);
- ❖ Lấy giá trị: session() -&gt;get(key);
- ❖ Hiển thị toàn bộ session trên hệ thống: session() -&gt;all();
- ❖ Xóa session: session()-&gt;forget (key);
- ❖ Xóa toàn bộ session trên hệ thống: session() -&gt;flush();

## Cookie

- ❖ Nên sử dụng các hàm thao tác với cookie mà Laravel cung cấp, thay vì sử dụng biến $\_COOKIE của PHP
- ❖ Set giá trị:
- ❖ $cookie = cookie('name', '123', '30');
- ❖ Cookie::queue($cookie);
- ❖ Lấy giá trị:
- ❖ Cookie::get('name');

## Views -Blade Template

- ❖ Vị trí nằm tại resources/views
- ❖ Laravel sử dụng engine blade cho view, với đuôi file là .blade.php
- ❖ Cú pháp trong view blade:
- ❖ Hiển thị dữ liệu: {{ &lt;tên -biến&gt; }} , VD: {{ $abc }}
- ❖ Các cấu trúc điều khiển
- ❖ @if -@else -@endif, @foreach - @endforeach, @for -@endfor, @while @endwhile
- ❖ Viết code PHP trong file blade: @php(&lt;code -php&gt;)
- ❖ VD: @php($a = 5; $a++)

## View – – Layout

- ❖ Thường được khai báo tại resource/views/layouts
- ❖ Ví dụ tạo layouts resource/views/layouts/master.blade.php
- ❖ @yield: có thể hiểu như 1 tham số, khi views nào extends từ layouts này thì lúc đó mới set giá trị thực

## View – – Layout

- ❖ Tạo views kế thừa layout sử dụng từ khóa @extends
- ❖ Ví dụ: tạo view child.blade.php extends từ layouts master.blade.php từ ví dụ trước
- ❖ @section: cú pháp tại lớp con để set giá trị, hay có thể hiểu là nó thay thế cú pháp @yield từ bên layout
- @extends('layouts.master')
- @section('title', 'This is title')
- @section('content')
- Content của tôi
- @endsection

## Redirect

- ❖ Chuyển hướng người dùng dựa theo route
- ❖ Có các phương thức sau
- ❖ redirect('&lt;tên-route&gt;');
- ❖ VD: chuyển hướng về link /home
- ❖ redirect('home');
- ❖ redirect()-&gt;route('&lt;route\_name&gt;');
- ❖ Với cách này cần set name cho route

## Làm việc với Database

- ❖ Giới thiệu về DB -Laravel
- ❖ Truy vấn QueryBuilder -Eloquent

## Giới thiệu về DB -Laravel

- ❖ Laravel cung cấp 2 cơ chế thao tác với Database
- ❖ QueryBuilder: gần với truy vấn thô, tốc độ nhanh hơn, sử dụng class façade DB theo hướng PHP thuần
- ❖ Eloquent ORM: cách viết đẹp và rõ ràng hơn, có tính bảo mật cao hơn, sử dụng Model theo hướng MVC
- ❖ Tùy mục đích mà có thể sử dụng cơ chế nào cho phù hợp
- ❖ Tạo model với artisan: php artisan make:model &lt;tên-model&gt;
- ❖ VD: php artisan make:model News

## Truy vấn QueryBuilder -Eloquent

```
QueryBuilder Eloquent ORM Lấy tất cả bản ghi $news = DB::table('news')->get(); $news = News::all() Lấy 1 bản ghi theo id $new = DB::table('news')->where('id', 1)->first(); $new = News::find(1); Lấy 1 trường của 1 bản ghi $new = DB::table('news')->where("id", 1)->select('title')->first(); $new = News::find(1)>select('title')->first(); $new = News::where('title', 'Hello World')->first(); Lấy 1 trường của tất cả bản ghi $new = DB::table('news')>select("title")->get(); $news = News::all('title');
```

## Truy vấn QueryBuilder -Eloquent

```
NTTrung Bài giảng CNWeb 23/25 QueryBuilder Eloquent ORM Insert bản ghi $isInsert = DB::table('news')>insert([ 'title' => 'new' ]); $newModel = new News(); $newModel ->title = "Title insert"; $newModel ->save(); Update bản ghi $isUpdate = DB::table('news')>where('id', 1)->update([ 'title' => 'title new' ]); $news = News::find(1); $news ->title = "Title update"; $news ->save(); Xóa bản ghi $isDelete = DB::table('news')->where('id', 1)>delete(); $news = News::find(1); $news ->delete(); Hàm count và max $count = DB::table('news')->count(); $max = DB::table('news')->max('id'); $count = News::count(); $max = News::max('id');
```

## Validate Form

- ❖ Laravel xử lý validate dựa vào class Request hoặc phương thức request()
- ❖ Form trong Laravel bắt buộc phải có trường ẩn sau, để tránh lỗi bảo mật CSRF
- ❖ &lt;input type="hidden" name="\_token" value="{{ csrf\_token() }}" /&gt;
- ❖ Xử lý validate trong Laravel tại Controller
- ❖ Ví dụ sau validate username và password không được để trống, username chỉ cho phép độ dài lớn nhất là 8 ký tự
- public function validateForm(Request $request) { $this-&gt;validate($request, ['username' =&gt; 'required|max:8', 'password' =&gt; 'required'

]);

}

![Image](/content/drive/MyDrive/MD/CNWeeb/8-CNWeb-PHP-Laravel_artifacts/image_000005_f8b3991a0a269ac977340db1ee9c7b9aebc8ac6ee41cd01e023f49619798f5eb.png)

![Image](/content/drive/MyDrive/MD/CNWeeb/8-CNWeb-PHP-Laravel_artifacts/image_000006_48f30b045eb7506d75204cb631fc95fcfbea776e914653268a29a4fa5fb67dff.png)