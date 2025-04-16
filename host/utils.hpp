#ifndef __UTILS_H__
#define __UTILS_H__
#include "typedef.hpp"
#include "debug_utils.hpp"
#include <math.h>
#include <chrono>


#define header_print(header, msg) \
    do { \
        std::ostringstream oss; \
        oss << msg; \
        std::cout << '[' << header << "]  " << oss.str() << std::endl; \
    } while (0)

namespace time_utils {

typedef std::chrono::high_resolution_clock::time_point time_point;
typedef std::pair<float, std::string> time_with_unit;

time_point now(){
    return std::chrono::high_resolution_clock::now();
}

time_with_unit duration_us(time_point start, time_point stop){
    return std::make_pair(std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count(), "us");
}

time_with_unit duration_ms(time_point start, time_point stop){
    return std::make_pair(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count(), "ms");
}

time_with_unit duration_s(time_point start, time_point stop){
    return std::make_pair(std::chrono::duration_cast<std::chrono::seconds>(stop - start).count(), "s");
}

time_with_unit cast_to_us(time_with_unit time){
    if (time.second == "us"){
        return time;
    }
    if (time.second == "ms"){
        return std::make_pair(time.first * 1000, "us");
    }
    if (time.second == "s"){
        return std::make_pair(time.first * 1000000, "us");
    }
    else{
        return time;
    }
}

time_with_unit cast_to_ms(time_with_unit time){
    if (time.second == "ms"){
        return time;
    }
    if (time.second == "us"){
        return std::make_pair(time.first / 1000, "ms");
    }
    if (time.second == "s"){
        return std::make_pair(time.first / 1000000, "ms");
    }
    else{
        return time;
    }
}   

time_with_unit cast_to_s(time_with_unit time){
    if (time.second == "s"){
        return time;
    }
    if (time.second == "us"){
        return std::make_pair(time.first / 1000000, "s");
    }
    if (time.second == "ms"){
        return std::make_pair(time.first / 1000, "s");
    }
    else{
        return time;
    }
}   

time_with_unit re_unit(time_with_unit time){
    time = cast_to_us(time);
    float time_us = time.first;
    std::string time_unit = time.second;
    if (time_us > 1000){
        time_us /= 1000;
        time_unit = "ms";
    }
    if (time_us > 1000000){
        time_us /= 1000000;
        time_unit = "s";
    }
    return std::make_pair(time_us, time_unit);
}


}

namespace utils {
float getRand(float min = 0, float max = 1){
    float a = (float)(rand()) / (float)(RAND_MAX) * (max - min) + min;
    if (abs(a) < 0.01){
        a = 0;
    }
    return a;
}

int getRandInt(int min = 0, int max = 6){
    return (int)(rand()) % (max - min) + min;
}

void print_progress_bar(std::ostream &os, double progress, int len = 75) {
    os  << "\r" << std::string((int)(progress * len), '|')
        << std::string(len - (int)(progress * len), ' ') << std::setw(4)
        << std::fixed << std::setprecision(0) << progress * 100 << "%"
        << "\r";
}

void check_arg_file_exists(std::string name) {
    // Attempt to open the file
    std::ifstream file(name);

    // Check if the file was successfully opened
    if (!file) {
        throw std::runtime_error("Error: File '" + name + "' does not exist or cannot be opened.");
    }

    // Optionally, close the file (not strictly necessary, as ifstream will close on destruction)
    file.close();
}

template <typename T>
bool nearly_equal(T a, T b, float abs_tol, float rel_tol){
    if (abs_tol == 0 && rel_tol == 0){
        return a == b;
    }
    else{
        // abs tol check
        bool equal = true;
        float err = abs((float)a - (float)b);

        if (abs_tol < 0){
            abs_tol = FLT_MAX;
        }
        
        if (err > abs_tol){
            equal = false;
            LOG_VERBOSE(1, "ABS");
        }

        float max = abs((float)a);
        if (abs((float)b) > max){
            max = abs((float)b);
        }

        float rel_err = max * rel_tol;

        if (err > rel_err){
            equal = false;
            LOG_VERBOSE(1, "REL");
        }
        LOG_VERBOSE_IF(1, !equal, "Error between " << a << " and " << b << " is " << err);
        LOG_VERBOSE_IF(1, !equal, "REL_ERROR " << rel_err);

        return equal;
    }
}

template <typename T>
int compare_vectors(vector<T>& y, vector<T>& y_ref, int print_errors = 16, float abs_tol = -1, float rel_tol = 0){
    int total_errors = 0;
    for (int i = 0; i < y.size(); i++){
        if (!nearly_equal(y[i], y_ref[i], abs_tol, rel_tol)){
            if (total_errors < print_errors){
                std::cout << "Error: y[" << i << "] = " << y[i] << " != y_ref[" << i << "] = " << y_ref[i] << std::endl;
            }
            total_errors++;
        }
    }
    return total_errors;
}

void print_npu_profile(time_utils::time_with_unit npu_time, float op){
    time_utils::time_with_unit time_united = time_utils::re_unit(npu_time);
    time_united = time_utils::cast_to_s(time_united);
    

    float ops = op / (time_united.first);
    float speed = ops / 1000000;
    std::string ops_unit = "Mops";
    std::string speed_unit = "Mops/s";
    if (speed > 1000){
        speed /= 1000;
        speed_unit = "Gops/s";
    }
    if (speed > 1000000){
        speed /= 1000000;
        speed_unit = "Tops/s";
    }

    time_united = time_utils::re_unit(npu_time);
    MSG_BONDLINE(40);
    MSG_BOX_LINE(40, "NPU time : " << time_united.first << " " << time_united.second);
    MSG_BOX_LINE(40, "NPU speed: " << speed << " " << speed_unit);
    MSG_BONDLINE(40);
}

void box_print(std::string msg, int width = 40){
    MSG_BOX(width, msg);
}

void box_print_bound(int width = 40){
    MSG_BONDLINE(width);
}

void box_print_line(std::string msg, int width = 40){
    MSG_BOX_LINE(width, msg);
}

template <typename T>
void print_matrix(const vector<T> matrix, int n_cols,
                  int n_printable_rows = 10, int n_printable_cols = 10,
                  std::ostream &ostream = std::cout,
                  const char col_sep[] = "  ", const char elide_sym[] = " ... ",
                  int w = -1) {
  assert(matrix.size() % n_cols == 0);

  if (w == -1) {
    w = 6;
  }
  int n_rows = matrix.size() / n_cols;

  n_printable_rows = std::min(n_rows, n_printable_rows);
  n_printable_cols = std::min(n_cols, n_printable_cols);

  const bool elide_rows = n_printable_rows < n_rows;
  const bool elide_cols = n_printable_cols < n_cols;

  if (elide_rows || elide_cols) {
    w = std::max((int)w, (int)strlen(elide_sym));
  }

  w += 3; // for decimal point and two decimal digits
  ostream << std::fixed << std::setprecision(2);

#define print_row(what)                                                        \
  for (int col = 0; col < (n_printable_cols + 1) / 2; col++) {                 \
    ostream << std::right << std::setw(w) << std::scientific << std::setprecision(2) << (what);                           \
    ostream << std::setw(0) << col_sep;                                        \
  }                                                                            \
  if (elide_cols) {                                                            \
    ostream << std::setw(0) << elide_sym;                                      \
  }                                                                            \
  for (int i = 0; i < n_printable_cols / 2; i++) {                             \
    int col = n_cols - n_printable_cols / 2 + i;                               \
    ostream << std::right << std::setw(w) << std::scientific << std::setprecision(2) << (what);                           \
    ostream << std::setw(0) << col_sep;                                        \
  }

  for (int row = 0; row < (n_printable_rows + 1) / 2; row++) {
    print_row(matrix[row * n_cols + col]);
    ostream << std::endl;
  }
  if (elide_rows) {
    print_row(elide_sym);
    ostream << std::endl;
  }
  for (int i = 0; i < n_printable_rows / 2; i++) {
    int row = n_rows - n_printable_rows / 2 + i;
    print_row(matrix[row * n_cols + col]);
    ostream << std::endl;
  }
#undef print_row
}
}
#endif
