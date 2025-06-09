#pragma once
#include "../layers/Layer.hpp"
// We only need forward declarations here, not full includes, which is cleaner
#include <vector>
#include <memory>
#include <string>

class DenseLayer;
template<int, int> class ConvolutionLayer;
class BatchNormLayer;


class Model {
public:
    // ... add, forward, backward, set_mode, save, load, predict are fine as declarations ...
    void add(std::unique_ptr<Layer> layer);
    Tensor forward(const Tensor& input);
    void backward(const Tensor& loss_gradient);
    void set_mode(bool is_training);
    void save(const std::string& filepath);
    void load(const std::string& filepath);
    std::vector<int> predict(const Tensor& input);

    // --- DECLARATIONS ONLY ---
    std::vector<Tensor*> get_parameters();
    std::vector<Tensor*> get_parameter_gradients();

private:
    std::vector<std::unique_ptr<Layer>> m_layers;
};