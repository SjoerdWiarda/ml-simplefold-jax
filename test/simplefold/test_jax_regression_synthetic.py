import jax

jax.config.update("jax_traceback_filtering", "off")
jax.config.update("jax_default_device", "cpu")
import numpy
import torch
from esm.modules import TransformerLayer as TransformerLayerTorch
from flax import nnx

from simplefold.model.jax.architecture import FoldingDiT as FoldingDiTJax
from simplefold.model.jax.blocks import DiTBlock as DiTBlockJax
from simplefold.model.jax.blocks import HomogenTrunk as HomogenTrunkJax
from simplefold.model.jax.esm_modules import TransformerLayer as TransformerLayerJAX
from simplefold.model.jax.esm_network import ESM2
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
from simplefold.utils.esm_utils import esm_model_dict
from simplefold.utils.jax_utils import replace_by_torch_dict, unflatten_state_dict


def test_folding_dit_regression():
    "Tests that the DiT Folding model has the same inference between Torch and JAX on synthetic data."

    num_heads = 2
    hidden_size = 12
    depth = 2
    esm_model = "esm2_8M"  # "esm2_3B"
    torch.manual_seed(42)

    jax.config.update("jax_enable_x64", True)

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
        key=jax.random.key(1), shape=(n_batch, n_tok, n_mol)
    )
    esm_s_dim = esm_model_dict[esm_model]["esm_s_dim"]
    esm_num_layers = esm_model_dict[esm_model]["esm_num_layers"]
    feats = {
        "ref_pos": jax.random.normal(key=jax.random.key(2), shape=(n_batch, n_tok, 3)),
        "mol_type": jax.random.randint(
            key=jax.random.key(3),
            minval=0,
            maxval=3,
            shape=(n_batch, n_mol),
            dtype=jax.numpy.int64,
        ),
        "atom_to_token": atom_to_token,
        "atom_to_token_idx": jax.numpy.argmax(atom_to_token, axis=-1),
        "ref_space_uid": jax.random.normal(
            key=jax.random.key(4), shape=(n_batch, n_tok)
        ),
        "ref_charge": jax.random.normal(key=jax.random.key(5), shape=(n_batch, n_tok)),
        "atom_pad_mask": jax.random.normal(
            key=jax.random.key(6), shape=(n_batch, n_tok)
        ),
        "max_num_tokens": 5,
        "res_type": jax.random.normal(
            key=jax.random.key(7), shape=(n_batch, n_mol, 33)
        ),
        "pocket_feature": jax.random.normal(
            key=jax.random.key(8), shape=(n_batch, n_mol, 4)
        ),
        "ref_element": jax.random.normal(
            key=jax.random.key(9), shape=(n_batch, n_tok, 128)
        ),
        "ref_atom_name_chars": jax.random.normal(
            key=jax.random.key(10), shape=(n_batch, n_tok, 256)
        ),
        "residue_index": jax.random.normal(
            key=jax.random.key(11), shape=(n_batch, n_mol)
        ),
        "entity_id": jax.random.normal(key=jax.random.key(12), shape=(n_batch, n_mol)),
        "asym_id": jax.random.normal(key=jax.random.key(13), shape=(n_batch, n_mol)),
        "sym_id": jax.random.normal(key=jax.random.key(14), shape=(n_batch, n_mol)),
        "esm_s": jax.random.normal(
            key=jax.random.key(15), shape=(n_batch, n_mol, esm_num_layers, esm_s_dim)
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


def test_transformer_regression() -> None:
    "Tests that the transformerlayer of the ESM model has the same inference Torch and JAX on synthetic data."

    numpy.random.seed(42)
    torch.manual_seed(42)
    jax.config.update("jax_enable_x64", True)

    attention_heads = 2
    head_dim = 32
    embed_dim = attention_heads * head_dim
    ffn_embed_dim = 4
    add_bias_kv = False
    use_esm1b_layer_norm = True
    use_rotary_embeddings = True

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
    assert numpy.isclose(torch_output[0].detach().numpy(), jax_output[0]).all()
