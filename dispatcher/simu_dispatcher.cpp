#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include <queue>
#include <string>
#include <algorithm>

const size_t total_jobs = 100;
size_t top_k = 20;
size_t predict_success_rate = 30;
const size_t actual_count_for = 70;

std::mt19937 rand_eng(42u);
std::uniform_int_distribution<uint32_t> rand_total_it(100, 500);

const size_t paralle_count = 2;
std::uniform_int_distribution<uint32_t> rand_mem_usage(10, 50);

struct Job
{
    uint32_t total_iterations;
    uint32_t actual_iterations;
    uint32_t predict_iterations; // for earyly-stopping
    uint32_t already_run_it;
    std::vector<uint32_t> mem_usage;
    uint32_t priority;
};

std::vector<Job> generate_jobs()
{
    std::vector<Job> ret;
    for (int i = 0; i < total_jobs; ++i)
    {
        uint32_t total_it = rand_total_it(rand_eng);

        uint32_t actual_min = total_it - total_it * (100 - actual_count_for) / 100;
        std::uniform_int_distribution<uint32_t> rand_actual_it(actual_min, total_it);
        uint32_t actual_it = rand_actual_it(rand_eng);

        uint32_t predict_min = actual_it - actual_it * (100 - predict_success_rate) / 100;
        uint32_t predict_max = actual_it + actual_it * (100 - predict_success_rate) / 100;
        std::uniform_int_distribution<uint32_t> rand_predic_it(predict_min, predict_max);
        uint32_t predict_it = rand_predic_it(rand_eng);

        std::vector<uint32_t> mem_usage;
        for (int j = 0; j < total_it; ++j)
        {
            mem_usage.push_back(rand_mem_usage(rand_eng));
        }

        Job j{
            .total_iterations = total_it,
            .actual_iterations = actual_it,
            .predict_iterations = predict_it,
            .already_run_it = 0,
            .mem_usage = std::move(mem_usage),
            .priority = static_cast<uint32_t>(total_jobs - i),
        };

        ret.push_back(std::move(j));
    }

    return ret;
}

void schedule_A(std::vector<Job> jobs)
{
    uint32_t now_it = 0;
    uint32_t turnaround_time = 0;
    uint32_t turnaround_time_with_weight = 0;
    uint32_t waiting_time = 0;
    std::vector<Job> running_jobs;
    while (running_jobs.size() < paralle_count && !jobs.empty())
    {
        running_jobs.push_back(*jobs.begin());
        jobs.erase(jobs.begin());
    }

    while (!running_jobs.empty() || !jobs.empty())
    {
        ++now_it;
        for (auto it = running_jobs.begin(); it != running_jobs.end();)
        {
            (it->already_run_it)++;
            if (it->already_run_it >= it->actual_iterations)
            {
                turnaround_time += now_it;
                turnaround_time_with_weight += (now_it * it->priority);
                it = running_jobs.erase(it);
            }
            else
            {
                ++it;
            }
        }
        while (running_jobs.size() < paralle_count && !jobs.empty())
        {
            running_jobs.push_back(*jobs.begin());
            jobs.erase(jobs.begin());
            waiting_time += now_it;
        }
    }

    std::cout << "" << turnaround_time / total_jobs
              << " " << turnaround_time_with_weight / ((1 + total_jobs) * total_jobs / 2)
              << " " << waiting_time / total_jobs
              << " " << now_it << std::endl;
}

void schedule_B(std::vector<Job> jobs)
{
    uint32_t now_it = 0;
    uint32_t turnaround_time = 0;
    uint32_t turnaround_time_with_weight = 0;
    uint32_t waiting_time = 0;
    std::vector<Job> running_jobs;
    while (running_jobs.size() < top_k && !jobs.empty())
    {
        running_jobs.push_back(*jobs.begin());
        jobs.erase(jobs.begin());
    }

    while (!running_jobs.empty() || !jobs.empty())
    {
        ++now_it;
        // schedule_for k
        std::sort(running_jobs.begin(), running_jobs.end(), [](const Job &l, const Job &r)
                  {
            if (l.predict_iterations == r.predict_iterations) {
                return l.mem_usage < r.mem_usage;
            }
            return l.predict_iterations < r.predict_iterations; });
        uint32_t mem_cnt = 0;
        for (auto it = running_jobs.begin(); it != running_jobs.end();)
        {
            mem_cnt += it->mem_usage[it->already_run_it];
            if (mem_cnt > 100)
            {
                break;
            }
            if (it->already_run_it == 0)
            {
                waiting_time += now_it;
            }

            (it->already_run_it)++;

            if (it->already_run_it >= it->actual_iterations)
            {
                turnaround_time += now_it;
                turnaround_time_with_weight += (now_it * it->priority);
                it = running_jobs.erase(it);
            }
            else
            {
                ++it;
            }
        }

        while (running_jobs.size() < paralle_count && !jobs.empty())
        {
            running_jobs.push_back(*jobs.begin());
            jobs.erase(jobs.begin());
        }
    }

    std::cout << "" << turnaround_time / total_jobs
              << "," << turnaround_time_with_weight / ((1 + total_jobs) * total_jobs / 2)
              << "," << waiting_time / total_jobs
              << "," << now_it << std::endl;
}

void schedule_C(std::vector<Job> jobs)
{
    uint32_t now_it = 0;
    uint32_t turnaround_time = 0;
    uint32_t turnaround_time_with_weight = 0;
    uint32_t waiting_time = 0;
    std::vector<Job> running_jobs(jobs);

    while (!running_jobs.empty())
    {
        ++now_it;
        using job_sort_t = std::pair<uint32_t, std::vector<Job>::iterator>;
        std::vector<job_sort_t> buffer;
        for (auto it = running_jobs.begin(); it != running_jobs.end(); ++it)
        {
            buffer.push_back(std::make_pair(it->mem_usage[it->already_run_it], it));
        }
        std::sort(buffer.begin(), buffer.end(), [](const job_sort_t &l, const job_sort_t &r)
                  { return l.first < r.first; });
        uint32_t mem = 0;
        for (auto it = buffer.begin(); it != buffer.end(); ++it)
        {
            if (it->first + mem > 100)
            {
                break;
            }
            mem += it->first;

            if (it->second->already_run_it == 0)
            {
                waiting_time += now_it;
            }
            it->second->already_run_it++;
            if (it->second->already_run_it >= it->second->actual_iterations)
            {
                turnaround_time += now_it;
                turnaround_time_with_weight += (now_it * it->second->priority);
                running_jobs.erase(it->second);
            }
        }
    }

    std::cout << "" << turnaround_time / total_jobs
              << " " << turnaround_time_with_weight / ((1 + total_jobs) * total_jobs / 2)
              << " " << waiting_time / total_jobs
              << " " << now_it << std::endl;
}

int main(int argc, char *argv[])
{
    top_k = std::atoi(argv[1]);
    predict_success_rate = std::atoi(argv[2]);

    auto jobs = generate_jobs();
    schedule_A(jobs);
    schedule_B(jobs);
    schedule_C(jobs);
    return 0;
}