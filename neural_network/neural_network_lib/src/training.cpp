#include "training.h"

Training::Training
(
    NeuralNetwork& kNeuralNetwork,
    const double kLearningRate,
    const double kMomentum,
    const std::string& kFilePath, 
    const size_t kInputsNumber, 
    const size_t kOutputsNumber, 
    const size_t kTrainingSetSize,
    const LossFunctionType& kType
) :
    m_neural_network(kNeuralNetwork, kLearningRate, kMomentum), m_training_file( std::ifstream(kFilePath, std::ios::in) ),
    m_inputs_number(kInputsNumber), m_outputs_number(kOutputsNumber), m_training_set_size(kTrainingSetSize)
{
    if (!m_training_file.is_open())
    {
        std::cerr << "Error! Training file was not opened!\n";
        exit(1);
    }
    ReadFile();
    m_loss_function = LossFunctionsFabric::CreateLossFunction(kType);
}

size_t Training::GetInputsNumber() const
{
    return m_inputs_number;
}

size_t Training::GetOutputsNumber() const
{
    return m_outputs_number;
}

size_t Training::GetTrainingSetSize() const
{
    return m_training_set_size;
}


Matrix<double> Training::GetInputMatrix() const
{
    return m_input_matrix;
}

Matrix<double> Training::GetOutputMatrix() const
{
    return m_targets_matrix;
}

void Training::ReadFile()
{
    FileHander fh;
    Matrix<double> read_matrix = fh.ReadMatrixFromFile(&m_training_file, 0, m_training_set_size, m_inputs_number+m_outputs_number);
    m_input_matrix = read_matrix.GetMtx(0, 0, read_matrix.GetRowsNum()-1, m_inputs_number - 1);
    m_targets_matrix = read_matrix.GetMtx(0, m_inputs_number, read_matrix.GetRowsNum()-1, read_matrix.GetColsNum()-1);
}

void Training::TrainOnSet()
{
    for (size_t it = 0; it < m_training_set_size; ++it)
    {
        Vector<double> input_data = m_input_matrix.GetRow(it);
        Vector<double> target_values = m_targets_matrix.GetRow(it);

        m_neural_network.AdjustmentNeuralNetwork(input_data, target_values);
    }
}

void Training::TrainNeuralNetwork(const size_t& kEpochNumber)
{
    for (int it = 0; it < kEpochNumber; ++it)
    {
        TrainOnSet();
        if (it % 50 == 0)
        {
            std::cout << "########## Epoch " << it << " ##########" << std::endl;
            DisplayResults();
            std::cout << "################################\n\n";
        }
    }
	std::cout << "###### End of training ######\n";
	DisplayResults();
	std::cout << "\n\n";
}

double Training::CalculateError()
{
    Matrix<double> actual_outputs(m_training_set_size, m_outputs_number);
    for (size_t it = 0; it < m_training_set_size; ++it)
    {
        Vector<double> input_data = m_input_matrix.GetRow(it);
        Vector<double> outputs = m_neural_network.GetNeuralNetwork().CalculateOutputs(input_data);
        actual_outputs.SetRow(it, outputs.GetVector());
    }
    return m_loss_function->CalculateError(m_targets_matrix, actual_outputs);
}

void Training::DisplayResults()
{
    std::stringstream out;
    out << "Neural network calculations:\n";
    for (size_t it = 0; it < m_training_set_size; ++it)
    {
        Vector<double> input_data = m_input_matrix.GetRow(it);
        Vector<double> outputs = m_neural_network.GetNeuralNetwork().CalculateOutputs(input_data);
        out << "Input data: " << input_data << "Targer data:" << m_targets_matrix.GetRow(it)
            << "Output data:" << outputs << "\n";
    }
    out << "Error:" << CalculateError() << "\n\n";
    std::cout << out.str();
}
