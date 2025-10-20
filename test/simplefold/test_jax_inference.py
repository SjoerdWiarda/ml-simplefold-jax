import jax

jax.config.update("jax_traceback_filtering", "off")
jax.config.update("jax_default_device", "cpu")

import argparse

from simplefold import __version__
from simplefold.inference import predict_structures_from_fastas
from simplefold.wrapper import InferenceWrapper, ModelWrapper


def test_inference() -> None:
    "Tests that all models used for inference are loaded succesfully using real data and that inference can be done with the JAX backend."

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
    plddt = True  # whether to use pLDDT confidence module
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
    assert set(results.keys()) == {"sampled_coord", "pad_mask", "plddts"}
    assert isinstance(results["sampled_coord"], jax.Array)
    assert jax.numpy.isfinite(results["sampled_coord"]).all()
    assert results["sampled_coord"].shape == (1, 2912, 3)

    assert isinstance(results["pad_mask"], jax.Array)
    assert jax.numpy.isfinite(results["pad_mask"]).all()
    assert results["pad_mask"].shape == (1, 2912)
    assert (
        jax.numpy.unique(results["pad_mask"]) == jax.numpy.asarray([0.0, 1.0])
    ).all()

    assert isinstance(results["plddts"], jax.Array)
    assert jax.numpy.isfinite(results["plddts"]).all()
    assert results["plddts"].shape == (1, 2912)


def test_inference_py() -> None:
    parser = argparse.ArgumentParser(
        prog="simplefold", description="Folding proteins with SimpleFold."
    )
    parser.add_argument(
        "--simplefold_model",
        type=str,
        default="simplefold_100M",
        help="Name of the model to load.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="artifacts",
        help="Directory to save the checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/debug_samples",
        help="Directory to save the output structure.",
    )
    parser.add_argument(
        "--num_steps", type=int, default=500, help="Number of steps in inference."
    )
    parser.add_argument(
        "--tau", type=float, default=0.1, help="Diffusion coefficient scaling factor."
    )
    parser.add_argument(
        "--no_log_timesteps", action="store_true", help="Disable logarithmic timesteps."
    )
    parser.add_argument(
        "--fasta_path",
        required=True,
        type=str,
        help="Path to the input FASTA file/directory.",
    )
    parser.add_argument(
        "--nsample_per_protein",
        type=int,
        default=1,
        help="Number of samples to generate per protein.",
    )
    parser.add_argument("--plddt", action="store_true", help="Enable pLDDT prediction.")
    parser.add_argument(
        "--output_format",
        type=str,
        default="mmcif",
        choices=["pdb", "mmcif"],
        help="Output file format.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="jax",
        choices=["torch", "mlx", "jax"],
        help="Backend to run inference either torch or mlx",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    args = parser.parse_args()
    predict_structures_from_fastas(args)

    # glob for f"{record.id}_sampled_{i}"
