#pragma once
#include<vector>

void init_matrix_case(std::vector<double>& mat,
                      std::vector<double>& vec,
                      int n);

void matrix_dot_naive(const std::vector<double>& mat,
                      const std::vector<double>& vec,
                      std::vector<double>& sum,
                      int n);

void matrix_dot_cache_opt(const std::vector<double>& mat,
                          const std::vector<double>& vec,
                          std::vector<double>& sum,
                          int n);

bool vectors_close(const std::vector<double>& a,
                   const std::vector<double>& b,
                   double eps = 1e-9);