//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/profiler.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <unordered_map>
#include <utility>

#include "duckdb/common/chrono.hpp"
#include "duckdb/common/helper.hpp"

namespace duckdb {

//! The profiler can be used to measure elapsed time
template <typename T>
class BaseProfiler {
public:
	//! Starts the timer
	void Start() {
		finished = false;
		start = Tick();
	}
	//! Finishes timing
	void End() {
		end = Tick();
		finished = true;
	}

	//! Returns the elapsed time in seconds. If End() has been called, returns
	//! the total elapsed time. Otherwise returns how far along the timer is
	//! right now.
	double Elapsed() const {
		auto _end = finished ? end : Tick();
		return std::chrono::duration_cast<std::chrono::duration<double>>(_end - start).count();
	}

private:
	time_point<T> Tick() const {
		return T::now();
	}
	time_point<T> start;
	time_point<T> end;
	bool finished = false;
};

using Profiler = BaseProfiler<system_clock>;

class BeeProfiler {
public:
	const static bool kEnableProfiling = true;

public:
	static BeeProfiler &Get() {
		static BeeProfiler instance;
		return instance;
	}

	void InsertStatRecord(string name, double value) {
		InsertStatRecord(name, uint64_t(value * 1e9));
	}

	inline void InsertStatRecord(string name, uint64_t value) {
		if (kEnableProfiling) {
			std::lock_guard<std::mutex> lock(mtx);
			values_[name] += value;
			calling_times_[name] += 1;
		}
	}

	void InsertHTRecord(string name, uint64_t tuple_sz, uint64_t point_table_sz, uint64_t num_terms) {
		if (kEnableProfiling) {
			std::lock_guard<std::mutex> lock(mtx);
			if (ht_records_.count(name) == 0) {
				ht_records_[name] = HTInfo(tuple_sz, point_table_sz, num_terms);
			}
		}
	}

	void EndProfiling() {
		if (kEnableProfiling) {
			PrintResults();
			Clear();
		}
	}

	void PrintResults() const {
		std::lock_guard<std::mutex> lock(mtx);

		// -------------------------------- Print Timing Results --------------------------------
		std::vector<std::string> keys;
		for (const auto &pair : values_) {
			keys.push_back(pair.first);
		}
		if (!keys.empty()) {
			std::sort(keys.begin(), keys.end());
			std::cerr << "-------\n";
			for (const auto &key : keys) {
				if (key.find("TableScan") != std::string::npos && key.find("in_mem") == std::string::npos) {
					continue;
				}
				if (key.find("#Tuple") != std::string::npos) {
					continue;
				}
				double time = values_.at(key) / double(1e9);
				uint64_t calling_times = calling_times_.at(key);
				double avg = time / calling_times;

				std::cerr << "Total: " << time << " s\tCalls: " << calling_times << "\tAvg: " << avg << " s\t" << key
				          << '\n';
			}
			std::cerr << "-------\n";
			for (const auto &key : keys) {
				if (key.find("#Tuple") != std::string::npos) {
					uint64_t total_tuples = values_.at(key);
					uint64_t calling_times = calling_times_.at(key);
					double avg = total_tuples / double(calling_times);

					std::cerr << "Total: " << total_tuples << "\tCalls: " << calling_times << "\tAvg: " << avg << "\t"
					          << key << '\n';
				}
			}
		}

		// -------------------------------- Print HT Results --------------------------------
		std::vector<string> ht_keys;
		for (const auto &pair : ht_records_) {
			ht_keys.push_back(pair.first);
		}
		if (!ht_keys.empty()) {
			std::cerr << "-------\n";
			std::sort(ht_keys.begin(), ht_keys.end());
			for (const auto &key : ht_keys) {
				auto ht_info = ht_records_.at(key);

				std::cerr << "Tuples Size: " << (double)ht_info.tuple_size / (1 << 20) << " MB\t"
				          << "Point Size: " << (double)ht_info.point_table_size / (1 << 20) << " MB\t"
				          << "#Term: " << ht_info.num_terms << "\t" << key << '\n';
			}
		}
	}

	void Clear() {
		std::lock_guard<std::mutex> lock(mtx);

		values_.clear();
		calling_times_.clear();
		ht_records_.clear();
	}

private:
	struct HTInfo {
		uint64_t tuple_size;
		uint64_t point_table_size;
		uint64_t num_terms;

		HTInfo(uint64_t ts = 0, uint64_t pts = 0, uint64_t nt = 0)
		    : tuple_size(ts), point_table_size(pts), num_terms(nt) {
		}
	};

	std::unordered_map<string, uint64_t> values_;
	std::unordered_map<string, uint64_t> calling_times_;
	std::unordered_map<string, HTInfo> ht_records_;
	mutable std::mutex mtx;
};

class CatProfiler {
public:
	static CatProfiler &Get() {
		static CatProfiler instance;
		return instance;
	}

	void StartStage(const string &stage_name) {
		std::lock_guard<std::mutex> lock(mtx_);
		if (!cur_stage_.empty() && stage_name == cur_stage_) {
			// already in this stage
			return;
		}
		cur_stage_ = stage_name;
		timer_.Start();
	}

	void EndStage(const string &stage_name) {
		std::lock_guard<std::mutex> lock(mtx_);
		if (stage_name != cur_stage_) {
			// not the same stage
			return;
		}
		timer_.End();
		stage_timings_[stage_name] += timer_.Elapsed();
		cur_stage_.clear();
	}

	std::unordered_map<string, double> &GetStageTimings() {
		if (!cur_stage_.empty()) {
			EndStage(cur_stage_);
		}
		return stage_timings_;
	}

	void PrintResults() const {
		for (const auto &pair : stage_timings_) {
			std::cerr << pair.first << ": " << pair.second << " s\t";
		}
	}

	void Clear() {
		stage_timings_.clear();
		timer_.End();
		cur_stage_.clear();
	}

private:
	// [stage_name] -> total time (ns)
	std::unordered_map<string, double> stage_timings_;
	std::mutex mtx_;
	Profiler timer_;
	string cur_stage_;
};

class ZebraProfiler {
public:
	const static bool kEnableProfiling = false;

public:
	static ZebraProfiler &Get() {
		static ZebraProfiler instance;
		return instance;
	}

	inline void InsertRecord(string name, uint64_t key, double value) {
		InsertRecord(std::move(name), key, uint64_t(value * 1e9));
	}

	inline void InsertRecord(string name, uint64_t key, uint64_t value) {
		if (kEnableProfiling) {
			D_ASSERT(key <= 2048);

			if (hists_.count(name) == 0) {
				std::lock_guard<std::mutex> lock(mutex_);
				hists_[name] = Histogram();
			}

			// insert record into the histogram of [name]
			auto &hist = hists_[name];
			hist.values_[key] += value;
			hist.cnt_[key] += 1;
		}
	}

	inline void PrintResults() const {
		if (kEnableProfiling) {
			for (auto &pair : hists_) {
				const auto &name = pair.first;
				const auto &hist = pair.second;

				std::cerr << "-------\n";
				std::cerr << name << '\n';
				for (uint64_t i = 1; i <= 2048; ++i) {
					if (hist.cnt_[i].load() > 0) {
						std::cerr << i << ": " << hist.values_[i] / double(1e3) / hist.cnt_[i] << " us\t"
						          << hist.cnt_[i] << '\n';
					}
				}
			}
		}
	}

	// output to csv file, each hist has a file.
	inline void ToCSV() {
		if (kEnableProfiling) {
			std::string folder_name = "./zebra_log_0x" + std::to_string(RandomInteger());
			std::filesystem::create_directories(folder_name);
			for (auto &pair : hists_) {
				const auto &name = pair.first;
				const auto &hist = pair.second;

				// filter these filters
				if (name.find("FILTER") != string::npos) {
					continue;
				}

				string file_name = folder_name + "/" + name + ".csv";
				std::ofstream out(file_name);
				out << "key, value, cnt\n";
				for (uint64_t i = 1; i <= 2048; ++i) {
					if (hist.cnt_[i] == 0) continue;
					out << i << "," << hist.values_[i] / double(1e3) / hist.cnt_[i] << "," << hist.cnt_[i] << "\n";
				}
				out.close();
			}
		}
	}

	inline void Clear() {
		hists_.clear();
	}

private:
	inline uint64_t RandomInteger() {
		return integers(gen_);
	}

	struct Histogram {
		std::vector<std::atomic<uint64_t>> values_;  // in ns
		std::vector<std::atomic<uint64_t>> cnt_;

		Histogram() : values_(2048 + 1), cnt_(2048 + 1) {
		}
	};

	std::mutex mutex_;
	std::unordered_map<string, Histogram> hists_;

	// random
	std::mt19937 gen_;
	std::uniform_int_distribution<int> integers;
};

// This profiler is to compute the chunk factor of each hash join operator
class HashJoinProfiler {
public:
	const static bool kEnableProfiling = false;

public:
	static HashJoinProfiler &Get() {
		static HashJoinProfiler instance;
		return instance;
	}

	void InputChunk(uint64_t n_tuple, const string &join_addr) {
		if (!kEnableProfiling) return;
		if (n_tuple == 0) return;

		auto &info = GetJoinInfo(join_addr);
		// update cardinality
		++info.n_input_chunk;
		++info.dist_input_size[n_tuple - 1];
	}

	void OutputChunk(uint64_t n_input, uint64_t n_output, const string &join_addr) {
		if (!kEnableProfiling) return;
		if (n_output == 0) return;

		auto &info = GetJoinInfo(join_addr);
		// update cardinality
		++info.n_output_chunk;
		++info.dist_output_size[n_output - 1];

		// update chunk factor
		if (n_input < n_output) std::cerr << "Error\n";
		double factor = n_input / double(n_output);
		info.sum_chunk_factors += factor;
		++info.n_chunk_factor;
		info.chunk_factors.push_back(factor);
	}

	void PrintProfile() {
		for (const auto &pair : joins_) {
			const auto &join_name = pair.first;
			const auto &info = pair.second;

			uint64_t total_input = 0, total_output = 0;
			for (uint64_t i = 0; i < 2048; ++i) {
				total_input += (i + 1) * info.dist_input_size[i];
				total_output += (i + 1) * info.dist_output_size[i];
			}
			double avg_input_tuple = total_input / double(info.n_input_chunk);
			double avg_output_tuple = total_output / double(info.n_output_chunk);
			double chunk_factor = info.sum_chunk_factors / info.n_chunk_factor;

			std::cerr << join_name << "\tChunk Factor: " << chunk_factor << "\n";
			std::cerr << "\tInput -- " << "#Tuple: " << total_input << "\t" << "#Chunk: " << info.n_input_chunk << "\t"
			          << "Avg Size: " << avg_input_tuple << "\n";
			std::cerr << "\tOutput -- " << "#Tuple: " << total_output << "\t" << "#Chunk: " << info.n_output_chunk
			          << "\t" << "Avg Size: " << avg_output_tuple << "\n";
			std::cerr << "\tData: [";
			for (double factor : info.chunk_factors)
				std::cerr << factor << ", ";
			std::cerr << "]\n";
		}
	}

	void Clear() {
		joins_.clear();
	}

private:
	struct VectorizedJoinInfo {
		// cardinality
		uint64_t n_input_chunk;
		uint64_t n_output_chunk;
		std::vector<uint64_t> dist_input_size;
		std::vector<uint64_t> dist_output_size;

		// chunk factor
		double sum_chunk_factors;
		uint64_t n_chunk_factor;
		std::vector<double> chunk_factors;

		VectorizedJoinInfo()
		    : n_input_chunk(0),
		      n_output_chunk(0),
		      dist_input_size(2048, 0),
		      dist_output_size(2048, 0),
		      sum_chunk_factors(0),
		      n_chunk_factor(0) {
		}
	};

	VectorizedJoinInfo &GetJoinInfo(const string &address) {
		std::lock_guard<std::mutex> lock(mtx_);
		return joins_[address];
	}

	std::unordered_map<string, VectorizedJoinInfo> joins_;
	std::mutex mtx_;
};
}  // namespace duckdb
