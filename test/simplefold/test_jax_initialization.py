import jax

jax.config.update("jax_traceback_filtering", "off")
jax.config.update("jax_default_device", "cpu")

from flax import nnx

from simplefold.model.jax.architecture import FoldingDiT as FoldingDiTJax
from simplefold.model.jax.confidence_module import ConfidenceModule
from simplefold.model.jax.esm_modules import TransformerLayer as TransformerLayerJAX
from simplefold.model.jax.esm_network import ESM2
from simplefold.wrapper import InferenceWrapper, ModelWrapper


def test_load_model_wrapper() -> None:
    "Tests that the DiT folding and plddt model can be loaded from saved weights with JAX."

    simplefold_model = "simplefold_100M"  # choose from 100M, 360M, 700M, 1.1B, 1.6B, 3B
    backend = "jax"
    ckpt_dir = "artifacts"
    plddt = True  # whether to use pLDDT confidence module

    # initialize the folding model and pLDDT model
    model_wrapper = ModelWrapper(
        simplefold_model=simplefold_model,
        ckpt_dir=ckpt_dir,
        plddt=plddt,
        backend=backend,
    )
    folding_model = model_wrapper.from_pretrained_folding_model()
    plddt_model = model_wrapper.from_pretrained_plddt_model()

    assert isinstance(folding_model, FoldingDiTJax)
    assert set(plddt_model.keys()) == {"plddt_latent_module", "plddt_out_module"}
    assert isinstance(plddt_model["plddt_latent_module"], FoldingDiTJax)
    assert isinstance(plddt_model["plddt_out_module"], ConfidenceModule)


def test_inference_wrapper():
    "Tests that the ESM model can be succesfully constructed from saved weights with the JAX backend."

    simplefold_model = "simplefold_100M"  # choose from 100M, 360M, 700M, 1.1B, 1.6B, 3B
    backend = "jax"  # choose from ["mlx", "torch", "jax"]
    output_dir = "artifacts"
    prediction_dir = f"predictions_{simplefold_model}_{backend}"
    num_steps = 500  # number of inference steps for flow-matching
    tau = 0.05  # stochasticity scale
    nsample_per_protein = 1  # number of samples per protein
    device = jax.local_devices()[0]

    # initialize the inference module with inference configurations
    inference_wrapper = InferenceWrapper(
        output_dir=output_dir,
        prediction_dir=prediction_dir,
        num_steps=num_steps,
        tau=tau,
        nsample_per_protein=nsample_per_protein,
        device=device,
        backend=backend,
    )

    assert isinstance(inference_wrapper.esm_model, ESM2)


def test_abstract_eval() -> None:
    "Test that several easy to set up models can be constructed."
    nnx.eval_shape(lambda: TransformerLayerJAX(6, 4, 6, rngs=nnx.Rngs(0)))
    nnx.eval_shape(lambda: ESM2(36, 2560, 40, rngs=nnx.Rngs(0)))
