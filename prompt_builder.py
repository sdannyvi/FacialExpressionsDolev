from PIL import Image


def build_conversation(prompt_type, system_role, top_examples_df, query_image, classes_list):
    """
    prompt_type: str. "multi_user_label_only", "few_shot_chat_style", "single_user_examples_first", "single_user_query_first".
    system_role (bool): use a system role as context or not.
    top_examples_df: dataframe, contains top K retrieved images, with columns: file_path and true_label.
    query_image: A PIL Image object (the actual query image)
    classes_list (list[str]): All possible emotion classes
    """

    # sort classes list
    classes_list = sorted(classes_list)


    conversation = []
    images = []


    if prompt_type == "multi_user_label_only":
        # add system role if true
        if system_role:
            conversation.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": f"You are an expert in classifying emotions from facial expressions in images.\n"
                                             f"You will be given three example images with their corresponding emotion labels, followed by a query image.\n"
                                             f"Based on the examples provided, analyze the facial expression in the query image and classify the expression into one of the following emotions: "
                                             f"{', '.join(classes_list)}.\n"
                                             f"Respond with only one word: the emotion label."
                                             }
                ]
            })
            query_text = "Classify the emotion in the image."
        else:
            query_text = (f"Based on the examples provided, please analyze the emotion in this image and select the best match from "
                          f"the following options: {', '.join(classes_list)}. "
                          f"Respond with only one word: the emotion name.")

        # for each example, create a user turn with image and label
        for _,row in top_examples_df.iterrows():
            image_path = row['file_path']
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": row['true_label']}
                ]
            })
            images.append(Image.open(image_path).convert('RGB'))

        conversation.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query_text}
            ]
        })
        images.append(query_image)


    elif prompt_type == "few_shot":
        # add system role if true
        if system_role:
            conversation.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": f"You are an expert in classifying emotions from facial expressions in images.\n"
                                             f"You will be given three example images with their corresponding emotion labels, followed by a query image.\n"
                                             f"Based on the examples provided, analyze the facial expression in the query image and classify the expression into one of the following emotions: "
                                             f"{', '.join(classes_list)}.\n"
                                             f"Respond with only one word: the emotion label."
                                             }
                ]
            })

            query_text = "Classify the emotion in the image."
        else:
            query_text = (
                f"Based on the examples provided, please analyze the emotion in this image and select the best match from "
                f"the following options: {', '.join(classes_list)}. "
                f"Respond with only one word: the emotion name.")

        for _,row in top_examples_df.iterrows():
            image_path = row['file_path']
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Classify the emotion in the image."}
                ]
            })
            conversation.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": row['true_label']}
                ]
            })
            images.append(Image.open(image_path).convert('RGB'))

        conversation.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query_text}
            ]
        })
        images.append(query_image)

    elif prompt_type == "single_user_examples_first":
        # add system role if true
        if system_role:
            conversation.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": f"You are an expert in classifying emotions from facial expressions in images.\n"
                                             f"You will be given three example images with their corresponding emotion labels, followed by a query image.\n"
                                             f"Based on the examples provided, analyze the facial expression in the query image and classify the expression into one of the following emotions: "
                                             f"{', '.join(classes_list)}.\n"
                                             f"Respond with only one word: the emotion label."
                                             }
                ]
            })

            query_text = "Classify the emotion in the image."
        else:
            query_text = (
                f"Based on the examples provided, please analyze the emotion in this image and select the best match from "
                f"the following options: {', '.join(classes_list)}. "
                f"Respond with only one word: the emotion name.")

        content = []
        for _,row in top_examples_df.iterrows():
            image_path = row['file_path']
            content.append({"type": "image"})
            content.append({
                "type": "text",
                "text": f"Example: This image shows a person expressing the emotion: '{row['true_label']}'."
            })
            images.append(Image.open(image_path).convert('RGB'))
        # add query image and its question
        content.append({"type": "image"})
        content.append({"type": "text", "text": query_text})
        images.append(query_image)

        conversation.append({"role": "user", "content": content})

    elif prompt_type == "single_user_query_first":
        # add system role if true
        if system_role:
            conversation.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": f"You are an expert in classifying emotions from facial expressions in images.\n"
                                             f"You will be given three example images with their corresponding emotion labels, followed by a query image.\n"
                                             f"Based on the examples provided, analyze the facial expression in the query image and classify the expression into one of the following emotions: "
                                             f"{', '.join(classes_list)}.\n"
                                             f"Respond with only one word: the emotion label."
                                             }
                ]
            })

            query_text = "Classify the emotion in the image."
        else:
            query_text = (
                f"Based on the examples provided, please analyze the emotion in this image and select the best match from "
                f"the following options: {', '.join(classes_list)}. "
                f"Respond with only one word: the emotion name.")

        content = []
        content.append({"type": "image"})
        content.append({"type": "text", "text": query_text})
        images.append(query_image)

        for _,row in top_examples_df.iterrows():
            image_path = row['file_path']
            content.append({"type": "image"})
            content.append({
                "type": "text",
                "text": f"Example: This image shows a person expressing the emotion: '{row['true_label']}'."
            })
            images.append(Image.open(image_path).convert('RGB'))

        conversation.append({"role": "user", "content": content})

    return conversation, images




