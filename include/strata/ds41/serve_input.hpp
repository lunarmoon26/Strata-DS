#pragma once
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <poll.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>

namespace strata::ds41::serve {
// Linux resident pipe reader. The caller owns fd. STOP is immediate; other lines
// are queued. QUIT terminates the reader without requiring the writer to close
// stdin. Destruction also interrupts idle input, including fatal engine exits.
class Input {
public:
    explicit Input(int fd=STDIN_FILENO): fd_(fd) {}
    ~Input() { shutdown_.store(true);if(reader_.joinable()) reader_.join(); }
    Input(const Input&)=delete;
    Input& operator=(const Input&)=delete;
    std::atomic<bool> stop{false};

    void start() {
        reader_=std::thread([this] {
            std::string pending;
            size_t searched=0;
            bool done=false;
            while(!shutdown_.load() && !done) {
                pollfd source{fd_,POLLIN,0};
                const int ready=::poll(&source,1,50);
                if(ready<0) { if(errno==EINTR) continue;break; }
                if(!ready) continue;
                if(source.revents&POLLNVAL) break;
                char buffer[4096];
                const ssize_t bytes=::read(fd_,buffer,sizeof(buffer));
                if(bytes<0) { if(errno==EINTR || errno==EAGAIN) continue;break; }
                if(!bytes) { if(!pending.empty()) deliver(std::move(pending));break; }
                pending.append(buffer,size_t(bytes));
                for(;;) {
                    const auto newline=pending.find('\n',searched);
                    if(newline==std::string::npos) { searched=pending.size();break; }
                    std::string line=pending.substr(0,newline);
                    pending.erase(0,newline+1);searched=0;
                    if(!deliver(std::move(line))) { done=true;break; }
                }
            }
            std::lock_guard<std::mutex> lock(mutex_);
            eof_=true;changed_.notify_all();
        });
    }
    bool next(std::string& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock,[&] { return !lines_.empty() || eof_; });
        return take(out);
    }
    bool poll(std::string& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        return take(out);
    }
    bool ended() {
        std::lock_guard<std::mutex> lock(mutex_);
        return eof_ && lines_.empty();
    }
private:
    bool take(std::string& out) {
        if(lines_.empty()) return false;
        out=std::move(lines_.front());lines_.pop_front();return true;
    }
    bool deliver(std::string line) {
        if(!line.empty() && line.back()=='\r') line.pop_back();
        if(line=="STOP") { stop.store(true);return true; }
        const bool keep_reading=line!="QUIT";
        std::lock_guard<std::mutex> lock(mutex_);
        lines_.push_back(std::move(line));changed_.notify_one();return keep_reading;
    }
    int fd_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<std::string> lines_;
    bool eof_=false;
    std::atomic<bool> shutdown_{false};
    std::thread reader_;
};
}
