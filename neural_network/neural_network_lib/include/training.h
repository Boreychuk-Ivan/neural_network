#pragma once

#include "pch.h"

class Training
{
private:
    std::ifstream m_training_file;
    size_t m_inputs_number;
    size_t m_outputs_number;
    Matrix<double> m_input_matrix;
    Matrix<double> m_output_matrix;
public:
    Training() = delete;
    Training(const std::string& kFilePath, const size_t kInputsNumber, const size_t kOutputsNumber):
        m_training_file{ std::ifstream(kFilePath, std::ios::in) }, m_inputs_number(kInputsNumber), m_outputs_number(kOutputsNumber){};

    void ReadFile();

};