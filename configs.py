
vit_D1_E96_H1_R224_P16={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 96
    }, 
            'encoder':{
                        'depth': 1,
                        'em_size': 96,
                        'd_K': 96,
                        'd_V':96, 
                        'num_heads': 1, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 96,
                        'n_classes':10
    }
    }

vit_D1_E96_H4_R224_P16={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 96
    }, 
            'encoder':{
                        'depth': 1,
                        'em_size': 96,
                        'd_K': 96*4,
                        'd_V':96*4, 
                        'num_heads': 4, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 96,
                        'n_classes':10
    }
    }

vit_D4_E96_H1_R224_P16={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 96
    }, 
            'encoder':{
                        'depth': 4,
                        'em_size': 96,
                        'd_K': 96,
                        'd_V':96, 
                        'num_heads': 1, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 96,
                        'n_classes':10
    }
    }

vit_D8_E96_H1_R224_P16={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 96
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 96,
                        'd_K': 96,
                        'd_V':96, 
                        'num_heads': 1, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 96,
                        'n_classes':10
    }
    }

vit_D8_E96_H4_R224_P16={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 96
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 96,
                        'd_K': 96*4,
                        'd_V':96*4, 
                        'num_heads': 4, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 96,
                        'n_classes':10
    }
    }

vit_D8_E96_h4_R224_P16={ # D: Depth, E: Embedding, h:Head but without dimension expansion, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 96
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 96,
                        'd_K': 96,
                        'd_V':96, 
                        'num_heads': 4, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 96,
                        'n_classes':10
    }
    }

vit_D8_E96_H4_R224_P32={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 96
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 96,
                        'd_K': 96*4,
                        'd_V':96*4, 
                        'num_heads': 4, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 96,
                        'n_classes':10
    }
    }

vit_D8_E96_h4_R224_P32={ # D: Depth, E: Embedding, H:Head but without dimension expansion, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 96
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 96,
                        'd_K': 96,
                        'd_V':96, 
                        'num_heads': 4, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 96,
                        'n_classes':10
    }
    }

vit_D8_E384_H4_R224_P16={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 384
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 384,
                        'd_K': 384,
                        'd_V':384, 
                        'num_heads': 4, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 384,
                        'n_classes':10
    }
    }

mresnet_Conv1d_D1_E96_K3_R224_P16={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 96
    }, 
            'encoder':{
                        'depth': 1,
                        'em_size': 96,
                        'in_channels': 197,  # N+1 = (224/16)**2 + 1
                        'out_channels': 197, 
                        'kernel_size_group': (3,), # for Conv1D 
                        'stride_group': (1,),
                        'padding_group': (1,),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 96,
                        'n_classes':10
    }
    }

mresnet_Conv1d_D1_E96_K3333_R224_P16={'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 96
    }, 
            'encoder':{
                        'depth': 1,
                        'em_size': 96,
                        'in_channels': 197,  # N+1 = (224/16)**2 + 1
                        'out_channels': 197, 
                        'kernel_size_group': (3,3,3,3), # for Conv1D 
                        'stride_group': (1,1,1,1),
                        'padding_group': (1,1,1,1),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 96,
                        'n_classes':10
    }
    }

mresnet_Conv1d_D1_E96_K1357_R224_P16={'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 96
    }, 
            'encoder':{
                        'depth': 1,
                        'em_size': 96,
                        'in_channels': 197,  # N+1 = (224/16)**2 + 1
                        'out_channels': 197, 
                        'kernel_size_group': (1,3,5,7), # for Conv1D 
                        'stride_group': (1,1,1,1),
                        'padding_group': (0,1,2,3),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 96,
                        'n_classes':10
    }
    }

mresnet_Conv1d_D4_E96_K3_R224_P16={'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 96
    }, 
            'encoder':{
                        'depth': 4,
                        'em_size': 96,
                        'in_channels': 197,  # N+1 = (224/16)**2 + 1
                        'out_channels': 197, 
                        'kernel_size_group': (3,), # for Conv1D 
                        'stride_group': (1,),
                        'padding_group': (1,),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 96,
                        'n_classes':10
    }
    }

mresnet_Conv1d_D8_E96_K3_R224_P16={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 96
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 96,
                        'in_channels': 197,  # N+1 = (224/16)**2 + 1
                        'out_channels': 197, 
                        'kernel_size_group': (3,), # for Conv1D 
                        'stride_group': (1,),
                        'padding_group': (1,),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 96,
                        'n_classes':10
    }
    }

mresnet_Conv1d_D8_E96_K3333_R224_P16={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 96
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 96,
                        'in_channels': 197,  # N+1 = (224/16)**2 + 1
                        'out_channels': 197, 
                        'kernel_size_group': (3,3,3,3), # for Conv1D 
                        'stride_group': (1,1,1,1),
                        'padding_group': (1,1,1,1),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 96,
                        'n_classes':10
    }
    }

mresnet_Conv1d_D8_E96_K3333_R224_P32={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 96
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 96,
                        'in_channels': 50,  # N+1 = (224/32)**2 + 1
                        'out_channels': 50, 
                        'kernel_size_group': (3,3,3,3), # for Conv1D 
                        'stride_group': (1,1,1,1),
                        'padding_group': (1,1,1,1),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 96,
                        'n_classes':10
    }
    }
