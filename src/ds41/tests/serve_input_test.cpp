#include "strata/ds41/serve_input.hpp"
#include <chrono>
#include <cstdio>
#include <stdexcept>

using strata::ds41::serve::Input;
namespace {
void require(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
struct Pipe {
    int fd[2];
    Pipe() { if(::pipe(fd)) throw std::runtime_error("pipe"); }
    ~Pipe() { for(int f:fd) if(f>=0) ::close(f); }
    void send(const std::string& text) {
        size_t sent=0;
        while(sent<text.size()) {
            const auto bytes=::write(fd[1],text.data()+sent,text.size()-sent);
            if(bytes<0 && errno==EINTR) continue;
            require(bytes>0,"write pipe");sent+=size_t(bytes);
        }
    }
    void finish() { ::close(fd[1]);fd[1]=-1; }
};
}
int main() {
    try {
        {
            Pipe pipe;Input input(pipe.fd[0]);input.start();
            pipe.send("GEN 2 1,2\r\nSTOP\nGEN 3 1,2,3\nQUIT\n");
            std::string line;
            require(input.next(line) && line=="GEN 2 1,2","CRLF framing");
            require(input.next(line) && line=="GEN 3 1,2,3" && input.stop.load(),"STOP bypasses queue");
            require(input.next(line) && line=="QUIT","QUIT framing");
            require(!input.next(line) && input.ended(),"QUIT must end input with writer still open");
        }
        {
            Pipe pipe;const auto start=std::chrono::steady_clock::now();
            {
                Input input(pipe.fd[0]);input.start();pipe.send("IDLE\n");std::string line;
                require(input.next(line) && line=="IDLE","idle reader started");
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            require(std::chrono::steady_clock::now()-start<std::chrono::seconds(1),"idle reader did not stop at destruction");
        }
        {
            Pipe pipe;Input input(pipe.fd[0]);input.start();
            const std::string large="GEN 2 "+std::string(2<<20,'7');
            std::thread writer([&] { pipe.send(large);pipe.finish(); });
            std::string line;const bool received=input.next(line);writer.join();
            require(received && line==large,"chunked long line or final EOF line changed");
            require(!input.next(line),"EOF did not end input");
        }
        std::puts("PASS resident input: STOP, CRLF, long/partial lines, EOF, open-writer QUIT, idle destruction");
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what());return 1; }
}
