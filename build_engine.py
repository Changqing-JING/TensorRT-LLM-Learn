"""Build INT4 AWQ quantized TensorRT-LLM engine (comparable to q4_k_m GGUF)"""
from tensorrt_llm import BuildConfig
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import CalibConfig, QuantAlgo, QuantConfig

if __name__ == '__main__':
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    engine_dir = "./qwen2.5_1.5b_engine"

    # INT4 AWQ quantization (similar to q4_k_m)
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.W4A16_AWQ,  # 4-bit weights, 16-bit activations
    )

    # Calibration config for AWQ
    calib_config = CalibConfig(
        calib_dataset='cnn_dailymail',
        calib_batches=64,
        calib_max_seq_length=512,
    )

    # Build config - batch size 1
    build_config = BuildConfig(
        max_batch_size=1,
        max_input_len=256,
        max_seq_len=512,
    )

    print(f"Building INT4 AWQ quantized engine for: {model_name}")
    print(f"Quantization: {quant_config.quant_algo}")
    print("This will take a while (quantization + calibration + build)...")

    llm = LLM(
        model=model_name,
        quant_config=quant_config,
        calib_config=calib_config,
        build_config=build_config,
    )

    print(f"\nSaving engine to: {engine_dir}")
    llm.save(engine_dir)

    print("Done!")
