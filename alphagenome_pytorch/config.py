class AlphaGenomeConfig:
    model_type = "alphagenome"

    def __init__(
        self,
        dims=(768, 896, 1024, 1152, 1280, 1408, 1536),
        basepairs=5,
        dna_embed_width=15,
        num_organisms=2,
        transformer_kwargs=None,
        head_specs=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.dims = tuple(dims)
        self.basepairs = basepairs
        self.dna_embed_width = dna_embed_width
        self.num_organisms = num_organisms

        self.transformer_kwargs = transformer_kwargs or {
            "depth": 8,
            "heads": 8,
            "dim_head_qk": 128,
            "dim_head_v": 192,
            "dropout": 0.1,
            "ff_expansion_factor": 2.0,
            "max_positions": 8192,
            "dim_pairwise": 128,
            "pairwise_every_num_single_blocks": 2,
            "single_to_pairwise_heads": 32,
            "pool_size": 16,
        }

        self.head_specs = head_specs or {
            "human": {
                "num_tracks_1bp": 3165,
                "num_tracks_128bp": 2733,
                "num_tracks_contacts": 28,
                "num_splicing_contexts": 282,
                "hidden_dim_splice_juncs": 512
            },
            "mouse": {
                "num_tracks_1bp": 730,
                "num_tracks_128bp": 310,
                "num_tracks_contacts": 8,
                "num_splicing_contexts": 75,
                "hidden_dim_splice_juncs": 512
            },
        }

    def get_head_spec(self, organism):
        if organism not in self.head_specs:
            raise ValueError(f"Organism '{organism}' not found in head_specs.")
        return {
            **self.head_specs[organism]
        }
