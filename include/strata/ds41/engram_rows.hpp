// include/strata/ds41/engram_rows.hpp - the engram rows of one step, read without the OS file cache.
//
// The two engram tables are about 101 GB each and a step reads only 24 rows of each (256 weight bytes and 8 scale
// bytes per row). Plain preads would leave those pages in the OS file cache, where they push out the routed experts
// that M4 keeps there. So the rows are read with upstream's DirectFile (O_DIRECT, a thread pool, many reads in
// flight), as upstream reads its n-gram table. Where the file system refuses O_DIRECT (tmpfs, some overlays), the
// reader falls back to plain preads and says so once.
#pragma once

#include "strata/platform/direct_file.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::ds41 {

class EngramRows {
public:
    struct Table {
        std::string path;
        uint64_t weight_offset = 0;   ///< byte offset of row 0's weights; each row has `row_bytes` bytes
        uint64_t scale_offset = 0;    ///< byte offset of row 0's scales; each row has `scale_bytes` bytes
    };
    /// Opens every table. `max_rows` rows per table per call at most. io_threads: reads in flight per table
    /// (platform::DirectFile::set_threads; 0 = its default of 16). Prefill reads many rows and uses 64.
    EngramRows(const std::vector<Table>& tables, int max_rows, int row_bytes = 256, int scale_bytes = 8,
               int io_threads = 0);
    ~EngramRows();
    EngramRows(const EngramRows&) = delete;
    EngramRows& operator=(const EngramRows&) = delete;

    /// For every table t, rows ids[t][0..n_rows): writes the weights to w_out[t] + i * row_bytes and the scales to
    /// s_out[t] + i * scale_bytes. All reads of all tables are in flight together. A failed read disables this reader.
    void read(const std::vector<const int64_t*>& ids, int n_rows, const std::vector<uint8_t*>& w_out,
              const std::vector<uint8_t*>& s_out);

    /// True when every table is read with O_DIRECT
    bool direct() const { return direct_; }

private:
    struct Req {
        uint64_t aligned;   ///< first byte read (4 KiB aligned)
        uint32_t length;    ///< bytes read (4 or 8 KiB)
        uint32_t skip;      ///< offset of the wanted bytes inside the read
        uint32_t bytes;     ///< wanted bytes
        uint8_t* dst;
    };
    std::vector<Table> tables_;
    std::vector<std::unique_ptr<platform::DirectFile>> files_;
    std::vector<int> fds_;   ///< the fallback (plain pread) per table, -1 when the table is read with O_DIRECT
    int max_rows_, row_bytes_, scale_bytes_;
    bool direct_ = true;
    bool failed_ = false;
    uint8_t* buf_ = nullptr;   ///< aligned: one 8 KiB block per request
    std::vector<Req> reqs_;
};

}  // namespace strata::ds41
