#pragma once
#include <functional>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <stdexcept>

namespace hvrt {

class ThreadPool {
public:
    explicit ThreadPool(int n_threads) : stop_(false) {
        if (n_threads <= 0) n_threads = 1;
        workers_.reserve(static_cast<size_t>(n_threads));
        for (int i = 0; i < n_threads; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }

    // Submit callable → future<T>
    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>
    {
        using RetT = std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<RetT()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<RetT> fut = task->get_future();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stop_) throw std::runtime_error("ThreadPool is stopped");
            tasks_.emplace([task]{ (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

    int size() const { return static_cast<int>(workers_.size()); }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]{ return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_;
};

} // namespace hvrt
