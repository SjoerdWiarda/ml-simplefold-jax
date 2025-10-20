import jax

jax.config.update("jax_traceback_filtering", "off")
jax.config.update("jax_default_device", "cpu")
import numpy
import torch

from simplefold.wrapper import InferenceWrapper, ModelWrapper


def test_esm_regression_wrapper() -> None:
    """Tests that the esm model has the same output between JAX and torch when loading in a real model."""
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
            dtype="double",
        )

        esm_s = inference_wrapper.esm_model(
            tokens=esmaa,
            repr_layers=range(37),
            need_head_weights=False,
        )

        return esm_s

    esmaa = numpy.random.choice(23, size=(1, 353))
    torch_batch = get_esm_prediction("torch", torch.tensor(esmaa))
    jax_batch = get_esm_prediction("jax", jax.numpy.asarray(esmaa))

    assert set(jax_batch.keys()) == set(torch_batch.keys())
    assert (
        jax.numpy.isclose(
            jax.numpy.asarray(torch_batch["logits"].detach().numpy()),
            jax_batch["logits"],
            rtol=1e-3,
            atol=1e-3,
        )
    ).all()
    assert set(jax_batch["representations"].keys()) == set(
        torch_batch["representations"].keys()
    )

    for key in torch_batch["representations"].keys():
        assert (
            jax.numpy.isclose(
                jax.numpy.asarray(torch_batch["representations"][key].detach().numpy()),
                jax_batch["representations"][key],
                rtol=1e-4,
                atol=1e-4,
            )
        ).all()


def test_batch_regression():
    "Tests that the batch object used for inference is the same between jax and torch on real data."
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
            dtype="double",
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
