vit_D1_E512_H1_R224_P16={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 1,
                        'em_size': 512,
                        'd_K': 512,
                        'd_V':512, 
                        'num_heads': 1, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }

vit_D1_E512_H4_R224_P16={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 1,
                        'em_size': 512,
                        'd_K': 512, # 512/4=128=d_k
                        'd_V':512, 
                        'num_heads': 4, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }

vit_D4_E512_H1_R224_P16={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 4,
                        'em_size': 512,
                        'd_K': 512,
                        'd_V':512, 
                        'num_heads': 1, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }

vit_D4_E512_H4_R224_P16={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 4,
                        'em_size': 512,
                        'd_K': 512,
                        'd_V':512, 
                        'num_heads': 4, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }


vit_D8_E512_H1_R224_P16={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 512,
                        'd_K': 512,
                        'd_V':512, 
                        'num_heads': 1, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }

vit_D8_E512_H4_R224_P16={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 512,
                        'd_K': 512, # d_k=64
                        'd_V':512, 
                        'num_heads': 4, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }

vit_D8_E512_H4_R224_P32={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 512,
                        'd_K': 512,
                        'd_V':512, 
                        'num_heads': 4, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }

vit_D12_E512_H8_R224_P32={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 12,
                        'em_size': 512,
                        'd_K': 512,
                        'd_V':512, 
                        'num_heads': 8, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }

# small scale model 
vit_D4_E128_H1_R224_P32={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 128
    }, 
            'encoder':{
                        'depth': 4,
                        'em_size': 128,
                        'd_K': 128,
                        'd_V':128, 
                        'num_heads': 1, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 128,
                        'n_classes':10
    }
    }

vit_D4_E128_H4_R224_P32={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 128
    }, 
            'encoder':{
                        'depth': 4,
                        'em_size': 128,
                        'd_K': 128,
                        'd_V':128, 
                        'num_heads': 4, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 128,
                        'n_classes':10
    }
    }


vit_D8_E128_H1_R224_P32={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 128
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 128,
                        'd_K': 128,
                        'd_V':128, 
                        'num_heads': 1, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 128,
                        'n_classes':10
    }
    }

vit_D8_E128_H4_R224_P32={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 128
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 128,
                        'd_K': 128,
                        'd_V':128, 
                        'num_heads': 4, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 128,
                        'n_classes':10
    }
    }



#merged model
vit_merger_D4_E128_H1_R224_P32={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 128
    }, 
            'encoder':{
                        'depth': 4,
                        'em_size': 128,
                        'd_K': 128,
                        'd_V':128, 
                        'num_heads': 1, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 128,
                        'n_classes':10
    }
    }

vit_merger_D4_E128_H4_R224_P32={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 128
    }, 
            'encoder':{
                        'depth': 4,
                        'em_size': 128,
                        'd_K': 128,
                        'd_V':128, 
                        'num_heads': 4, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 128,
                        'n_classes':10
    }
    }



vit_merger_D8_E128_H1_R224_P32={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 128
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 128,
                        'd_K': 128,
                        'd_V':128, 
                        'num_heads': 1, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 128,
                        'n_classes':10
    }
    }

vit_merger_D8_E128_H4_R224_P32={ # D: Depth, E: Embedding, H:Head, R:Resolution, P:Patch
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 128
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 128,
                        'd_K': 128,
                        'd_V':128, 
                        'num_heads': 4, # d_K = heads * d_k
                        'att_drop_out': 0.,

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 128,
                        'n_classes':10
    }
    }






covit_D1_E512_K3_R224_P16={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 1,
                        'em_size': 512,
                        'kernel_size_group': (3,), # for Conv1D 
                        'stride_group': (1,),
                        'padding_group': (1,),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }

covit_D1_E512_K3333_R224_P16={'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 1,
                        'em_size': 512,
                        'kernel_size_group': (3,3,3,3), # for Conv1D 
                        'stride_group': (1,1,1,1),
                        'padding_group': (1,1,1,1),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }


covit_D4_E512_K3_R224_P16={'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 4,
                        'em_size': 512,
                        'kernel_size_group': (3,), # for Conv1D 
                        'stride_group': (1,),
                        'padding_group': (1,),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }

covit_D4_E512_K3333_R224_P16={'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 4,
                        'em_size': 512,
                        'kernel_size_group': (3,3,3,3), # for Conv1D 
                        'stride_group': (1,1,1,1),
                        'padding_group': (1,1,1,1),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }

covit_D4_E512_K7777_R224_P16={'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 4,
                        'em_size': 512,
                        'kernel_size_group': (7,7,7,7), # for Conv1D 
                        'stride_group': (1,1,1,1),
                        'padding_group': (3,3,3,3),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }


covit_D8_E512_K3_R224_P16={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 512,
                        'kernel_size_group': (3,), # for Conv1D 
                        'stride_group': (1,),
                        'padding_group': (1,),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }

covit_D8_E512_K1357_R224_P16={'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 512,
                        'kernel_size_group': (1,3,5,7), # for Conv1D 
                        'stride_group': (1,1,1,1),
                        'padding_group': (0,1,2,3),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }


covit_D8_E512_K3333_R224_P16={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 512,
                        'kernel_size_group': (3,3,3,3), # for Conv1D 
                        'stride_group': (1,1,1,1),
                        'padding_group': (1,1,1,1),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }

covit_D8_E512_K7777_R224_P16={'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (16, 16),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 512,
                        'kernel_size_group': (7,7,7,7), # for Conv1D 
                        'stride_group': (1,1,1,1),
                        'padding_group': (3,3,3,3),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }

covit_D8_E512_K3333_R224_P32={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 512,
                        'kernel_size_group': (3,3,3,3), # for Conv1D 
                        'stride_group': (1,1,1,1),
                        'padding_group': (1,1,1,1),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }

covit_D12_E512_4xK3_4xK5_R224_P32={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 12,
                        'em_size': 512,
                        'kernel_size_group': (3,3,3,3,5,5,5,5), # for Conv1D 
                        'stride_group': (1,1,1,1,1,1,1,1),
                        'padding_group': (1,1,1,1,2,2,2,2),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }

covit_D16_E512_4xK3_4xK5_R224_P32={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 512
    }, 
            'encoder':{
                        'depth': 16,
                        'em_size': 512,
                        'kernel_size_group': (3,3,3,3,5,5,5,5), # for Conv1D 
                        'stride_group': (1,1,1,1,1,1,1,1),
                        'padding_group': (1,1,1,1,2,2,2,2),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 512,
                        'n_classes':10
    }
    }


covit_D4_E128_K3_R224_P32={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 128
    }, 
            'encoder':{
                        'depth': 4,
                        'em_size': 128,
                        'kernel_size_group': (3,), # for Conv1D 
                        'stride_group': (1,),
                        'padding_group': (1,),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 128,
                        'n_classes':10
    }
    }


#small model for covit
covit_D4_E128_K3_R224_P32={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 128
    }, 
            'encoder':{
                        'depth': 4,
                        'em_size': 128,
                        'kernel_size_group': (3,), # for Conv1D 
                        'stride_group': (1,),
                        'padding_group': (1,),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 128,
                        'n_classes':10
    }
    }

covit_D4_E128_K3333_R224_P32={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 128
    }, 
            'encoder':{
                        'depth': 4,
                        'em_size': 128,
                        'kernel_size_group': (3,3,3,3), # for Conv1D 
                        'stride_group': (1,1,1,1),
                        'padding_group': (1,1,1,1),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 128,
                        'n_classes':10
    }
    }

covit_D8_E128_K3_R224_P32={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 128
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 128,
                        'kernel_size_group': (3,), # for Conv1D 
                        'stride_group': (1,),
                        'padding_group': (1,),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 128,
                        'n_classes':10
    }
    }

covit_D8_E128_K3333_R224_P32={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 128
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 128,
                        'kernel_size_group': (3,3,3,3), # for Conv1D 
                        'stride_group': (1,1,1,1),
                        'padding_group': (1,1,1,1),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 128,
                        'n_classes':10
    }
    }


#merged model for covit
covit_merger_D4_E128_K3_R224_P32={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 128
    }, 
            'encoder':{
                        'depth': 4,
                        'em_size': 128,
                        'kernel_size_group': (3,), # for Conv1D 
                        'stride_group': (1,),
                        'padding_group': (1,),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 128,
                        'n_classes':10
    }
    }

covit_merger_D4_E128_K3333_R224_P32={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 128
    }, 
            'encoder':{
                        'depth': 4,
                        'em_size': 128,
                        'kernel_size_group': (3,3,3,3), # for Conv1D 
                        'stride_group': (1,1,1,1),
                        'padding_group': (1,1,1,1),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 128,
                        'n_classes':10
    }
    }

covit_merger_D8_E128_K3_R224_P32={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 128
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 128,
                        'kernel_size_group': (3,), # for Conv1D 
                        'stride_group': (1,),
                        'padding_group': (1,),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 128,
                        'n_classes':10
    }
    }

covit_merger_D8_E128_K3333_R224_P32={ # D: Depth, E: Embedding, K:Kernel size, R:Resolution, P:Patch size
        'embedding':{
                        'in_channels': 3,
                        'img_size': (224, 224),
                        'patch_size': (32, 32),
                        'em_size': 128
    }, 
            'encoder':{
                        'depth': 8,
                        'em_size': 128,
                        'kernel_size_group': (3,3,3,3), # for Conv1D 
                        'stride_group': (1,1,1,1),
                        'padding_group': (1,1,1,1),

                        'MLP_expansion': 4,
                        'MLP_drop_out': 0. 

    },
            'cls_head':{
                        'em_size': 128,
                        'n_classes':10
    }
    }
