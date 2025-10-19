import os

os.environ["HYDRA_FULL_ERROR"] = "1"
from itertools import starmap
from typing import Any

import hydra
import jax

jax.config.update("jax_traceback_filtering", "off")
jax.config.update("jax_default_device", "cpu")
import numpy
import omegaconf
import torch
from flax import nnx

from simplefold.utils.jax_utils import replace_by_torch_dict, unflatten_state_dict
from simplefold.wrapper import InferenceWrapper, ModelWrapper, ckpt_url_dict


def test_load_checkpoint():
    simplefold_model = "simplefold_100M"  # choose from 100M, 360M, 700M, 1.1B, 1.6B, 3B
    ckpt_dir = "artifacts"

    # create folding model
    ckpt_path = os.path.join(ckpt_dir, f"{simplefold_model}.ckpt")
    if not os.path.exists(ckpt_path):
        os.system(f"curl -L -o {ckpt_path} {ckpt_url_dict[simplefold_model]}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # load model checkpoint
    cfg_path = os.path.join(
        "configs/model/architecture", f"foldingdit_{simplefold_model[11:]}.yaml"
    )

    # replace torch implementations with jax
    with open(cfg_path, "r") as f:
        yaml_str = f.read()
    yaml_str = yaml_str.replace("torch", "jax")

    model_config = omegaconf.OmegaConf.create(yaml_str)

    model = hydra.utils.instantiate(model_config)

    graphdef, state = nnx.split(model)
    second_unflattend_dict = unflatten_state_dict(checkpoint)

    updated_dict = replace_by_torch_dict(state, second_unflattend_dict)
    new_model = nnx.merge(graphdef, updated_dict)


def test_load_model_wrapper():

    simplefold_model = "simplefold_100M"  # choose from 100M, 360M, 700M, 1.1B, 1.6B, 3B
    backend = "jax"  # choose from ["mlx", "torch", "jax"]
    ckpt_dir = "artifacts"
    plddt = False  # whether to use pLDDT confidence module

    # initialize the folding model and pLDDT model
    model_wrapper = ModelWrapper(
        simplefold_model=simplefold_model,
        ckpt_dir=ckpt_dir,
        plddt=plddt,
        backend=backend,
    )
    folding_model = model_wrapper.from_pretrained_folding_model()
    plddt_model = model_wrapper.from_pretrained_plddt_model()


def test_inference_wrapper():

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


def test_inference():

    # following are example amino acid sequences:
    example_sequences = {
        "7ftv_A": "GASKLRAVLEKLKLSRDDISTAAGMVKGVVDHLLLRLKCDSAFRGVGLLNTGSYYEHVKISAPNEFDVMFKLEVPRIQLEEYSNTRAYYFVKFKRNPKENPLSQFLEGEILSASKMLSKFRKIIKEEINDDTDVIMKRKRGGSPAVTLLISEKISVDITLALESKSSWPASTQEGLRIQNWLSAKVRKQLRLKPFYLVPKHAEETWRLSFSHIEKEILNNHGKSKTCCENKEEKCCRKDCLKLMKYLLEQLKERFKDKKHLDKFSSYHVKTAFFHVCTQNPQDSQWDRKDLGLCFDNCVTYFLQCLRTEKLENYFIPEFNLFSSNLIDKRSKEFLTKQIEYERNNEFPVFD",
        "8cny_A": "MGPSLDFALSLLRRNIRQVQTDQGHFTMLGVRDRLAVLPRHSQPGKTIWVEHKLINILDAVELVDEQGVNLELTLVTLDTNEKFRDITKFIPENISAASDATLVINTEHMPSMFVPVGDVVQYGFLNLSGKPTHRTMMYNFPTKAGQCGGVVTSVGKVIGIHIGGNGRQGFCAGLKRSYFAS",
        "8g8r_A": "GTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFS",
        "8i85_A": "MGILQANRVLLSRLLPGVEPEGLTVRHGQFHQVVIASDRVVCLPRTAAAAARLPRRAAVMRVLAGLDLGCRTPRPLCEGSLPFLVLSRVPGAPLEADALEDSKVAEVVAAQYVTLLSGLASAGADEKVRAALPAPQGRWRQFAADVRAELFPLMSDGGCRQAERELAALDSLPDITEAVVHGNLGAENVLWVRDDGLPRLSGVIDWDEVSIGDPAEDLAAIGAGYGKDFLDQVLTLGGWSDRRMATRIATIRATFALQQALSACRDGDEEELADGLTGYR",
        "8g8r_A_x": "GTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFSGTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFSISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFSGTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFS",
    }
    seq_id = "7ftv_A"  # choose from example_sequences
    aa_sequence = example_sequences[seq_id]
    print(f"Predicting structure for {seq_id} with {len(aa_sequence)} amino acids.")

    simplefold_model = "simplefold_100M"  # choose from 100M, 360M, 700M, 1.1B, 1.6B, 3B
    backend = "jax"  # choose from ["mlx", "torch", "jax"]
    ckpt_dir = "artifacts"
    output_dir = "artifacts"
    prediction_dir = f"predictions_{simplefold_model}_{backend}"
    output_name = f"{seq_id}"
    num_steps = 500  # number of inference steps for flow-matching
    tau = 0.05  # stochasticity scale
    plddt = False  # whether to use pLDDT confidence module
    nsample_per_protein = 1  # number of samples per protein

    # initialize the folding model and pLDDT model
    model_wrapper = ModelWrapper(
        simplefold_model=simplefold_model,
        ckpt_dir=ckpt_dir,
        plddt=plddt,
        backend=backend,
    )
    device = model_wrapper.device
    folding_model = model_wrapper.from_pretrained_folding_model()
    plddt_model = model_wrapper.from_pretrained_plddt_model()

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

    # process input sequence and run inference
    batch, structure, record = inference_wrapper.process_input(aa_sequence)
    results = inference_wrapper.run_inference(
        batch,
        folding_model,
        plddt_model,
        device=device,
    )
    save_paths = inference_wrapper.save_result(
        structure, record, results, out_name=output_name
    )


from esm.modules import TransformerLayer as TransformerLayerTorch
from flax import nnx

from simplefold.model.jax.esm_modules import TransformerLayer as TransformerLayerJAX
from simplefold.model.jax.esm_network import ESM2
from simplefold.utils.esm_utils import _af2_to_esm, esm_registry


def test_transformer_regression() -> None:
    numpy.random.seed(42)
    torch.manual_seed(42)
    jax.config.update("jax_enable_x64", True)

    attention_heads = 2
    head_dim = 31
    embed_dim = attention_heads * head_dim
    ffn_embed_dim = 4
    add_bias_kv = True
    use_esm1b_layer_norm = False
    use_rotary_embeddings = False

    tfl_torch = (
        TransformerLayerTorch(
            embed_dim,
            ffn_embed_dim,
            attention_heads,
            add_bias_kv,
            use_esm1b_layer_norm,
            use_rotary_embeddings,
        )
        .eval()
        .double()
    )
    tfl_jax = nnx.eval_shape(
        lambda: TransformerLayerJAX(
            embed_dim,
            ffn_embed_dim,
            attention_heads,
            add_bias_kv,
            use_esm1b_layer_norm,
            use_rotary_embeddings,
            rngs=nnx.Rngs(0),
        )
    )

    esm_state_dict_torch = tfl_torch.cpu().state_dict()

    graphdef, state = nnx.split(tfl_jax)
    second_unflattend_dict = unflatten_state_dict(esm_state_dict_torch)

    # takes long
    updated_dict = replace_by_torch_dict(state, second_unflattend_dict)
    tfl_jax = nnx.merge(graphdef, updated_dict)
    tfl_jax.eval()
    # (T, B, E)
    x = numpy.random.random((3, 5, embed_dim))
    self_attn_mask = None
    self_attn_padding_mask = None
    need_head_weights = False

    torch_output = tfl_torch(
        torch.tensor(x, dtype=torch.float64),
        self_attn_mask,
        self_attn_padding_mask,
        need_head_weights,
    )
    jax_output = tfl_jax(
        jax.numpy.asarray(x), self_attn_mask, self_attn_padding_mask, need_head_weights
    )

    assert numpy.isclose(
        torch_output[0].detach().numpy(), jax_output[0]  # , atol=1e-4  # , rtol=1e-3
    ).all()


def test_folding_dit_regression():

    num_heads = 2
    hidden_size = 12
    depth = 2
    esm_model = "esm2_8M"  # "esm2_3B"
    torch.manual_seed(42)

    jax.config.update("jax_enable_x64", True)

    from simplefold.model.torch.architecture import FoldingDiT as FoldingDiTTorch
    from simplefold.model.torch.blocks import DiTBlock as DiTBlockTorch
    from simplefold.model.torch.blocks import HomogenTrunk as HomogenTrunkTorch
    from simplefold.model.torch.layers import (
        EfficientSelfAttentionLayer as EfficientSelfAttentionLayerTorch,
    )
    from simplefold.model.torch.layers import TimestepEmbedder as TimestepEmbedderTorch
    from simplefold.model.torch.pos_embed import (
        AbsolutePositionEncoding as AbsolutePositionEncodingTorch,
    )
    from simplefold.model.torch.pos_embed import (
        FourierPositionEncoding as FourierPositionEncodingTorch,
    )

    torch_dit_folder = FoldingDiTTorch(
        atom_hidden_size_enc=hidden_size,
        atom_hidden_size_dec=hidden_size,
        esm_model=esm_model,
        time_embedder=TimestepEmbedderTorch(hidden_size=hidden_size),
        hidden_size=hidden_size,
        aminoacid_pos_embedder=AbsolutePositionEncodingTorch(
            in_dim=1, embed_dim=hidden_size
        ),
        pos_embedder=FourierPositionEncodingTorch(in_dim=3),
        trunk=HomogenTrunkTorch(
            depth=depth,
            block=lambda: DiTBlockTorch(
                hidden_size=hidden_size,
                self_attention_layer=lambda: EfficientSelfAttentionLayerTorch(
                    hidden_size=hidden_size, num_heads=num_heads
                ),
            ),
        ),
        atom_encoder_transformer=HomogenTrunkTorch(
            depth=depth,
            block=lambda: DiTBlockTorch(
                hidden_size=hidden_size,
                self_attention_layer=lambda: EfficientSelfAttentionLayerTorch(
                    hidden_size=hidden_size, num_heads=num_heads
                ),
            ),
        ),
        atom_decoder_transformer=HomogenTrunkTorch(
            depth=depth,
            block=lambda: DiTBlockTorch(
                hidden_size=hidden_size,
                self_attention_layer=lambda: EfficientSelfAttentionLayerTorch(
                    hidden_size=hidden_size, num_heads=num_heads
                ),
            ),
        ),
    ).double()

    from simplefold.model.jax.architecture import FoldingDiT as FoldingDiTJax
    from simplefold.model.jax.blocks import DiTBlock as DiTBlockJax
    from simplefold.model.jax.blocks import HomogenTrunk as HomogenTrunkJax
    from simplefold.model.jax.layers import (
        EfficientSelfAttentionLayer as EfficientSelfAttentionLayerJax,
    )
    from simplefold.model.jax.layers import TimestepEmbedder as TimestepEmbedderJax
    from simplefold.model.jax.pos_embed import (
        AbsolutePositionEncoding as AbsolutePositionEncodingJax,
    )
    from simplefold.model.jax.pos_embed import (
        FourierPositionEncoding as FourierPositionEncodingJax,
    )
    from simplefold.utils.esm_utils import esm_model_dict

    folding_dit_jax = FoldingDiTJax(
        atom_hidden_size_enc=hidden_size,
        atom_hidden_size_dec=hidden_size,
        esm_model=esm_model,
        time_embedder=TimestepEmbedderJax(hidden_size=hidden_size),
        hidden_size=hidden_size,
        aminoacid_pos_embedder=AbsolutePositionEncodingJax(
            in_dim=1, embed_dim=hidden_size
        ),
        pos_embedder=FourierPositionEncodingJax(in_dim=3),
        trunk=HomogenTrunkJax(
            depth=depth,
            block=lambda: DiTBlockJax(
                hidden_size=hidden_size,
                self_attention_layer=lambda: EfficientSelfAttentionLayerJax(
                    hidden_size=hidden_size, num_heads=num_heads
                ),
            ),
        ),
        atom_encoder_transformer=HomogenTrunkJax(
            depth=depth,
            block=lambda: DiTBlockJax(
                hidden_size=hidden_size,
                self_attention_layer=lambda: EfficientSelfAttentionLayerJax(
                    hidden_size=hidden_size, num_heads=num_heads
                ),
            ),
        ),
        atom_decoder_transformer=HomogenTrunkJax(
            depth=depth,
            block=lambda: DiTBlockJax(
                hidden_size=hidden_size,
                self_attention_layer=lambda: EfficientSelfAttentionLayerJax(
                    hidden_size=hidden_size, num_heads=num_heads
                ),
            ),
        ),
    )

    n_batch = 3
    n_tok = 5
    n_mol = 3
    noised_pos = jax.random.normal(key=jax.random.key(0), shape=(n_batch, n_tok, 3))
    t = jax.numpy.linspace(0, 1, n_batch)

    atom_to_token = jax.random.normal(
        key=jax.random.key(2), shape=(n_batch, n_tok, n_mol)
    )
    esm_s_dim = esm_model_dict[esm_model]["esm_s_dim"]
    esm_num_layers = esm_model_dict[esm_model]["esm_num_layers"]
    feats = {
        "ref_pos": jax.random.normal(key=jax.random.key(0), shape=(n_batch, n_tok, 3)),
        "mol_type": jax.random.randint(
            key=jax.random.key(1),
            minval=0,
            maxval=3,
            shape=(n_batch, n_mol),
            dtype=jax.numpy.int64,
        ),
        "atom_to_token": atom_to_token,
        "atom_to_token_idx": jax.numpy.argmax(atom_to_token, axis=-1),
        "ref_space_uid": jax.random.normal(
            key=jax.random.key(0), shape=(n_batch, n_tok)
        ),
        "ref_charge": jax.random.normal(key=jax.random.key(0), shape=(n_batch, n_tok)),
        "atom_pad_mask": jax.random.normal(
            key=jax.random.key(0), shape=(n_batch, n_tok)
        ),
        "max_num_tokens": 5,
        "res_type": jax.random.normal(
            key=jax.random.key(1), shape=(n_batch, n_mol, 33)
        ),
        "pocket_feature": jax.random.normal(
            key=jax.random.key(1), shape=(n_batch, n_mol, 4)
        ),
        "ref_element": jax.random.normal(
            key=jax.random.key(1), shape=(n_batch, n_tok, 128)
        ),
        "ref_atom_name_chars": jax.random.normal(
            key=jax.random.key(1), shape=(n_batch, n_tok, 256)
        ),
        "residue_index": jax.random.normal(
            key=jax.random.key(1), shape=(n_batch, n_mol)
        ),
        "entity_id": jax.random.normal(key=jax.random.key(1), shape=(n_batch, n_mol)),
        "asym_id": jax.random.normal(key=jax.random.key(1), shape=(n_batch, n_mol)),
        "sym_id": jax.random.normal(key=jax.random.key(1), shape=(n_batch, n_mol)),
        "esm_s": jax.random.normal(
            key=jax.random.key(3), shape=(n_batch, n_mol, esm_num_layers, esm_s_dim)
        ),
    }

    esm_state_dict_torch = torch_dit_folder.cpu().state_dict()

    graphdef, state = nnx.split(folding_dit_jax)
    second_unflattend_dict = unflatten_state_dict(esm_state_dict_torch)

    # takes long
    updated_dict = replace_by_torch_dict(state, second_unflattend_dict)
    folding_dit_jax = nnx.merge(graphdef, updated_dict)
    folding_dit_jax.eval()

    torch_output = torch_dit_folder(
        noised_pos=torch.tensor(noised_pos),
        t=torch.tensor(t),
        feats={k: torch.tensor(v) for k, v in feats.items()},
    )
    jax_output = folding_dit_jax(noised_pos=noised_pos, t=t, feats=feats)

    assert torch_output.keys() == jax_output.keys()
    for key in torch_output.keys():
        assert numpy.isclose(
            torch_output[key].detach().numpy(),
            jax_output[key],
        ).all()


# def test_transformer_layer_regression() -> None:

#     backend = "jax"
#     esm_model, esm_dict = esm_registry["esm2_3B"]()  # should be 3B
#     af2_to_esm = _af2_to_esm(esm_dict)

#     if backend == "torch":
#         esm_model = esm_model.to(self.device)
#         af2_to_esm = af2_to_esm.to(self.device)
#         esm_model = esm_model.eval()
#     elif backend == "jax":
#         # takes long
#         esm_model_jax = ESM2(num_layers=36, embed_dim=2560, attention_heads=40)

#         esm_state_dict_torch = esm_model.cpu().state_dict()

#         graphdef, state = nnx.split(esm_model_jax)
#         second_unflattend_dict = unflatten_state_dict(esm_state_dict_torch)

#         # takes long
#         updated_dict = replace_by_torch_dict(state, second_unflattend_dict)
#         esm_model_jax = nnx.merge(graphdef, updated_dict)

#         # TODO: Force inference mode
#         esm_model = esm_model_jax
#     else:
#         raise NotImplementedError


def test_esm_regression_wrapper() -> None:
    """Tests that the"""
    numpy.random.seed(42)
    torch.manual_seed(42)
    jax.config.update("jax_enable_x64", True)

    def get_esm_prediction(backend, esmaa):
        # following are example amino acid sequences:
        simplefold_model = (
            "simplefold_100M"  # choose from 100M, 360M, 700M, 1.1B, 1.6B, 3B
        )
        ckpt_dir = "artifacts"
        output_dir = "artifacts"
        prediction_dir = f"predictions_{simplefold_model}_{backend}"
        num_steps = 500  # number of inference steps for flow-matching
        tau = 0.05  # stochasticity scale
        plddt = False  # whether to use pLDDT confidence module
        nsample_per_protein = 1  # number of samples per protein

        # initialize the folding model and pLDDT model
        model_wrapper = ModelWrapper(
            simplefold_model=simplefold_model,
            ckpt_dir=ckpt_dir,
            plddt=plddt,
            backend=backend,
        )
        device = model_wrapper.device

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

        esm_s = inference_wrapper.esm_model(
            tokens=esmaa,
            repr_layers=range(37),
            need_head_weights=False,
        )

        return esm_s

    esmaa = numpy.random.choice(23, size=(1, 353))
    jax_batch = get_esm_prediction("jax", jax.numpy.asarray(esmaa))
    torch_batch = get_esm_prediction("torch", torch.tensor(esmaa))

    assert set(jax_batch.keys()) == set(torch_batch.keys())
    for key in torch_batch.keys():
        if isinstance(torch_batch[key], torch.Tensor):
            assert (
                jax.numpy.isclose(
                    jax.numpy.asarray(torch_batch[key].detach().numpy()), jax_batch[key]
                )
            ).all()
            # TODO: check devices are the same.
        else:
            assert torch_batch[key] == jax_batch[key]


def test_abstract_eval() -> None:

    nnx.eval_shape(lambda: TransformerLayerJAX(6, 4, 6, rngs=nnx.Rngs(0)))
    nnx.eval_shape(lambda: ESM2(36, 2560, 40, rngs=nnx.Rngs(0)))


def test_batch_regression():
    jax.config.update("jax_enable_x64", True)

    def get_batch(backend):
        # following are example amino acid sequences:
        example_sequences = {
            "7ftv_A": "GASKLRAVLEKLKLSRDDISTAAGMVKGVVDHLLLRLKCDSAFRGVGLLNTGSYYEHVKISAPNEFDVMFKLEVPRIQLEEYSNTRAYYFVKFKRNPKENPLSQFLEGEILSASKMLSKFRKIIKEEINDDTDVIMKRKRGGSPAVTLLISEKISVDITLALESKSSWPASTQEGLRIQNWLSAKVRKQLRLKPFYLVPKHAEETWRLSFSHIEKEILNNHGKSKTCCENKEEKCCRKDCLKLMKYLLEQLKERFKDKKHLDKFSSYHVKTAFFHVCTQNPQDSQWDRKDLGLCFDNCVTYFLQCLRTEKLENYFIPEFNLFSSNLIDKRSKEFLTKQIEYERNNEFPVFD",
            "8cny_A": "MGPSLDFALSLLRRNIRQVQTDQGHFTMLGVRDRLAVLPRHSQPGKTIWVEHKLINILDAVELVDEQGVNLELTLVTLDTNEKFRDITKFIPENISAASDATLVINTEHMPSMFVPVGDVVQYGFLNLSGKPTHRTMMYNFPTKAGQCGGVVTSVGKVIGIHIGGNGRQGFCAGLKRSYFAS",
            "8g8r_A": "GTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFS",
            "8i85_A": "MGILQANRVLLSRLLPGVEPEGLTVRHGQFHQVVIASDRVVCLPRTAAAAARLPRRAAVMRVLAGLDLGCRTPRPLCEGSLPFLVLSRVPGAPLEADALEDSKVAEVVAAQYVTLLSGLASAGADEKVRAALPAPQGRWRQFAADVRAELFPLMSDGGCRQAERELAALDSLPDITEAVVHGNLGAENVLWVRDDGLPRLSGVIDWDEVSIGDPAEDLAAIGAGYGKDFLDQVLTLGGWSDRRMATRIATIRATFALQQALSACRDGDEEELADGLTGYR",
            "8g8r_A_x": "GTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFSGTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFSISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFSGTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFS",
        }
        seq_id = "7ftv_A"  # choose from example_sequences
        aa_sequence = example_sequences[seq_id]

        simplefold_model = (
            "simplefold_100M"  # choose from 100M, 360M, 700M, 1.1B, 1.6B, 3B
        )
        ckpt_dir = "artifacts"
        output_dir = "artifacts"
        prediction_dir = f"predictions_{simplefold_model}_{backend}"

        num_steps = 500  # number of inference steps for flow-matching
        tau = 0.05  # stochasticity scale
        plddt = False  # whether to use pLDDT confidence module
        nsample_per_protein = 1  # number of samples per protein

        # initialize the folding model and pLDDT model
        model_wrapper = ModelWrapper(
            simplefold_model=simplefold_model,
            ckpt_dir=ckpt_dir,
            plddt=plddt,
            backend=backend,
        )
        device = model_wrapper.device

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

        # process input sequence and run inference
        batch, structure, record = inference_wrapper.process_input(aa_sequence)
        return batch

    torch_batch = get_batch("torch")
    jax_batch = get_batch("jax")

    assert set(jax_batch.keys()) == set(torch_batch.keys())
    for key in torch_batch.keys():
        if isinstance(torch_batch[key], torch.Tensor):
            assert (
                jax.numpy.isclose(
                    jax.device_get(jax.numpy.asarray(torch_batch[key])),
                    jax_batch[key],
                    atol=1e-2,
                    rtol=1e-2,
                )
            ).all()
            # TODO: check devices are the same.
        else:
            assert torch_batch[key] == jax_batch[key]
