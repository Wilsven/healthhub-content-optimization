import logging
import re
import string
import unicodedata

from bs4 import BeautifulSoup, NavigableString, PageElement

# Set up logger in extractor.py
# Edit conf/logging.yml to see changes
logger = logging.getLogger(__name__)


class HTMLExtractor:
    """
    A class to extract and process various elements from HTML content
    using BeautifulSoup.

    Attributes:
        content_name (str): The name of the article
        content_category (str): The category of the article
        url (str): The URL of the article
        soup (BeautifulSoup): A BeautifulSoup object.
    """

    def __init__(
        self, content_name: str, content_category: str, full_url: str, html_content: str
    ) -> None:
        """
        Initializes the HTMLExtractor with the given HTML content.

        Args:
            content_name (str): The name of the article
            content_category (str): The category of the article
            full_url (str): The URL of the article
            html_content (str): The HTML content to be processed.
        """
        logger.debug(
            f"Text Extraction - Extracting `{content_name}` within `{content_category}`. Link to article - `{full_url}`"
        )

        self.content_name = content_name
        self.content_category = content_category
        self.url = full_url
        self.soup = self.preprocess_html(html_content)

        # Check how many direct children the HTML content has for debugging purposes
        num_children = len(list(self.soup.children))
        logger.debug(f"Text Extraction - {num_children} children detected")

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Cleans the given text by normalizing Unicode characters,
        handling special symbols, replacing problematic characters,
        and removing multiple whitespace.

        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: The cleaned text.

        Note:
            This method is being used in all extractor methods.
        """
        # Normalize Unicode characters
        text = unicodedata.normalize("NFKD", text)
        # Use ASCII encoding to handle special symbols e.g. copyright \xa9
        text = text.encode("ascii", "ignore").decode("utf-8")

        # Replace common problematic characters
        text = text.replace("\xa0", " ")  # non-breaking space
        text = text.replace("\u200b", "")  # zero-width space
        text = text.replace("\u2028", "\n")  # line separator
        text = text.replace("\u2029", "\n")  # paragraph separator
        text = text.replace("_x000D_", "")  # Carriage return

        # Replace multiple whitespace with single space
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    @classmethod
    def preprocess_html(cls, html_content: str) -> BeautifulSoup:
        """
        Preprocesses the given HTML content by replacing all <br>
        tags with newline characters.

        Args:
            html_content (str): The HTML content to be preprocessed.

        Returns:
            BeautifulSoup: The preprocessed HTML content as a BeautifulSoup object.
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Find all <br> tags and replace them with newline
        # NOTE: These logs are commented out as it is only used during development
        for br in soup.find_all("br"):
            br.replace_with("\n")
            # logger.debug("Text Extraction - Replacing br with newline")

        # Find all <hr> tags and replace them with newline
        for hr in soup.find_all("hr"):
            hr.replace_with("\n")
            # logger.debug("Text Extraction - Replacing hr with newline")

        return soup

    def extract_text(self) -> str:
        """
        Extracts the main content from the HTML content.

        Returns:
            str: The main content body extracted from the HTML content.

        Note:
            This function unwraps the HTML content if it is contained in a <div>. It then removes tables and extracts
            the main content by iterating over the tags in the soup. The following tags are considered:

                - h1, h2, h3, h4, h5, h6: These tags are treated as key headers and are paragraphed between them.
                - p: This tag is treated as a paragraph.
                    * If the text does not contain sentences about HealthHub app, Google Play, or Apple Store,
                    and it contains a strong tag, it is treated differently based on the text content.
                - ul: This tag is treated as an unordered list. If it is the child of a <div>, it is treated as a list.
                - ol: This tag is treated as an ordered list.
                - div: This tag is treated as a text within a div.
                - span: This tag is treated as a text within a span.
                - blockquote: This tag is treated as a text within a blockquote.

            The extracted content is stored in a list and then processed (_clean_up_fragments).
            Double newlines are replaced with single newlines and whitespace is stripped.
            If the processed text is empty, the function attempts to extract the content from the <div> tags.
        """
        # Unwrap if the HTML content is contained in a div
        # NOTE: The log is commented out as it is only used during development
        if self.soup.div is not None:
            # logger.debug("Text Extraction - Unwrapping outermost div container")
            self.soup.div.unwrap()

        # Remove all tables from the HTML text
        for table in self.soup.find_all("table"):
            table.extract()
            logger.debug(f"Text Extraction - Removing table from {self.content_name}")

        # Extract the main content
        content = []
        for tag in self.soup.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "p",
                "ul",
                "ol",
                "div",
                "span",
                "blockquote",
            ],
            recursive=False,
        ):
            # For texts within div and span - recursive search
            if tag.name in ["div", "span", "blockquote"]:
                # Changes content via Pass By Reference
                self._extract_text_from_container(tag, content)
            # Handles the text for the rest
            else:
                self._extract_text_elements(tag, content)

        # Clean up content fragments
        cleaned_content = self._clean_up_fragments(content)

        # Replace double newlines with single newlines and strip whitespace
        extracted_content_body = (
            "\n".join(cleaned_content).replace("\n\n", "\n").strip()
        )

        return extracted_content_body

    @classmethod
    def _clean_up_fragments(cls, content: list[str]) -> list[str]:
        """
        Cleans up content fragments after extraction

        Args:
            content (list[str]): The content to be cleaned.

        Returns:
            list[str]: The cleaned content fragments after removing spaces, joining sentences and bullet points
        """
        threshold = 3
        # Remove empty strings from content
        content = [c for c in content if c]

        result = []
        for i in range(len(content)):
            # Append the first element
            if i == 0:
                result.append(content[i])
            # Handle self-made bullet points
            elif result[-1] != "\n" and len(result[-1]) < threshold:
                result[-1] = result[-1] + " " + content[i]
            # Handle nested bullet points
            elif result[-1] == "\t":
                result[-1] = result[-1] + content[i]
            # Concatenate strings if the 2nd one starts with a punctuation
            elif content[i][0] in [".", ",", ":"]:
                result[-1] = result[-1] + content[i]
            else:
                result.append(content[i])

        return result

    def _extract_text_elements(self, tag: PageElement, content: list[str]) -> None:
        """
        Helper method to extract the text elements from the given HTML tag.

        This method extracts the text content from the various textual elements
        such as headers, paragraphs, anchors, lists, etc.

        Args:
            tag (PageElement): The HTML element to extract the text elements from.
            content (list[str]): A list of text elements.

        Returns:
            None: This method modifies the content list in-place.

        Note:
            This method is used in `extract_text` method and `_extract_text_from_container` method.
        """
        # For headings
        if tag.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            # Provide paragraphing between key headers
            content.append("\n")
            content.append(self.clean_text(tag.text))

        # For texts with strong importance
        elif tag.name == "strong":
            self._extract_text_from_strong(tag, content)

        # For paragraphs
        elif tag.name == "p":
            self._extract_text_from_p(tag, content)

        # Skip alternate texts in images - Extracted as a separate column instead
        elif tag.name == "img":
            pass

        elif tag.name == "em":
            self._extract_text_from_em(tag, content)

        # For unordered lists
        elif tag.name == "ul":
            self._extract_text_from_ul(tag, content)

        # For ordered lists
        elif tag.name == "ol":
            self._extract_text_from_ol(tag, content)

        # For texts in anchor tags (e.g. links)
        elif tag.name == "a":
            if len(content) > 0 and content[-1] != "\n":
                content[-1] = content[-1] + " " + self.clean_text(tag.text)
            else:
                content.append(self.clean_text(tag.text))

        # For lists
        elif tag.name == "li":
            content.append("- " + self.clean_text(tag.text))

        # For texts wrapped in b, u and i tags (bold, underlined and italicised)
        elif tag.name in ["b", "u", "i"]:
            cleaned_text = self.clean_text(tag.text)
            if len(content) == 0:
                content.append(cleaned_text)
            elif len(content) > 0 and cleaned_text not in ["", content[-1]]:
                content.append(cleaned_text)

        # To handle texts that have different fonts (wrapped in font tag)
        elif tag.name == "font":
            content[-1] = content[-1] + " " + self.clean_text(tag.text)

        # To handle texts that are wrapped as a superscript
        elif tag.name == "sup":
            content.append(self.clean_text(tag.text))

        # Skip tags
        elif tag.name in ["ins", "sub", "iframe", "style"]:
            pass

        # Monitor edge cases that are not handled
        else:
            cleaned_text = self.clean_text(tag.text)
            if self.content_category in [
                "cost-and-financing",
                "diseases-and-conditions",
                "live-healthy-articles",
                "medical-care-and-facilities",
                "support-group-and-others",
            ]:
                logger.debug(
                    f"Text Extraction - Tag {tag.name} not handled within {self._extract_text_elements.__name__}: {cleaned_text[:25]}"
                )

    def _extract_text_from_strong(self, tag: PageElement, content: list[str]) -> None:
        """
        Helper method to extract the text content from the strong element.

        Args:
            tag (PageElement): The HTML element to extract the text elements from.
            content (list[str]): A list of text elements.

        Returns:
            None: This method modifies the content list in-place.

        Note:
            This method is used in `extract_text` method.
        """
        # Find related section within strong
        cleaned_text = self.clean_text(tag.text)
        if "Related:" in cleaned_text:
            cleaned_text = re.sub(r"\n", " ", cleaned_text)
            if len(content) > 0 and cleaned_text != content[-1]:
                content.append(cleaned_text)
        elif "Read these next:" in cleaned_text:
            if len(content) > 0 and cleaned_text != content[-1]:
                content.append("\n")
                content.append(cleaned_text)
        else:
            for child in tag.children:
                # Return to _extract_text_element in if nested tags are detected
                if child.name in ["strong", "p", "a", "em"]:
                    self._extract_text_elements(child, content)
                # Return to _extract_text_element in if unordered list is detected
                elif child.name in ["ul"]:
                    self._extract_text_elements(child, content)
                # Extract text within strong
                elif child.name is None:
                    # Join text with previous element if punctuation is detected
                    if cleaned_text != "" and cleaned_text[0] in string.punctuation:
                        content[-1] = content[-1] + cleaned_text
                    elif cleaned_text != "":
                        content.append(cleaned_text)
                # For img tags
                elif child.name == "img":
                    self._extract_text_elements(child, content)
                # Add back underline words
                elif child.name in ["u"]:
                    cleaned_text = self.clean_text(child.text)
                    if cleaned_text != "" and len(content) > 0:
                        content[-1] = content[-1] + cleaned_text
                # Extract text in span containers
                elif child.name in ["span"]:
                    self._extract_text_from_container(child, content)
                # Skip tags
                elif child.name in ["sup"]:
                    continue
                # Monitor for missed edge cases
                else:
                    logger.debug(
                        f"Text Extraction - Tag {child.name} not handled within {tag.name}: {child.text[:25]}"
                    )

    def _extract_text_from_p(self, tag: PageElement, content: list[str]) -> None:
        """
        Helper method to extract the text content from the p element.

        Args:
            tag (PageElement): The HTML element to extract the text elements from.
            content (list[str]): A list of text elements.

        Returns:
            None: This method modifies the content list in-place.

        Note:
            This method is used in `extract_text` method.
        """
        # Skip sentences about HealthHub app, Google Play, and Apple Store
        if re.search(r"(HealthHub app|Google Play|Apple Store|Parent Hub)", tag.text):
            return
        else:
            for child in tag.children:
                # Return to _extract_text_element in if nested tags are detected
                if child.name in ["strong", "a", "h2", "ul", "p"]:
                    self._extract_text_elements(child, content)
                # Extract text that are emphasised
                elif child.name == "em":
                    # Note: Text can be either words or paragraphs - Impact formatting
                    cleaned_text = self.clean_text(child.text)
                    if cleaned_text != "":
                        content.append(self.clean_text(cleaned_text))
                # For img
                elif child.name == "img":
                    self._extract_text_elements(child, content)
                # Extract text in span containers
                elif child.name == "span":
                    self._extract_text_from_container(child, content)
                # Add back words that are bold
                elif child.name == "b":
                    cleaned_text = self.clean_text(child.text)
                    if len(content) == 0:
                        content.append(cleaned_text)
                    elif len(content) > 0 and cleaned_text not in ["", content[-1]]:
                        content.append(cleaned_text)
                # Add back words that are underlined or italicised
                elif child.name in ["u", "i"]:
                    # FIXME: Image credit is not being removed from text (To be reviewed again)
                    cleaned_text = self.clean_text(child.text)
                    # Skip texts with "Image credit"
                    if "Image credit" in cleaned_text:
                        continue
                    elif cleaned_text != "":
                        content.append(cleaned_text)
                # Extract text within p
                elif child.name is None:
                    cleaned_text = self.clean_text(child.text)
                    if len(content) == 0:
                        content.append(cleaned_text)
                    elif len(content) > 0 and cleaned_text not in ["", content[-1]]:
                        content.append(cleaned_text)
                    elif len(content) > 0 and cleaned_text in string.punctuation:
                        content[-1] = content[-1] + cleaned_text
                # Skip tags
                # Note: text in sup tags are skipped as they are mainly numbers
                elif child.name in [
                    "sub",
                    "sup",
                    "font",
                    "iframe",
                    "style",
                    "figure",
                ]:
                    continue
                else:
                    # TODO: Monitor for edge cases & Resolve it
                    logger.debug(
                        f"Text Extraction - Tag {child.name} not handled within {tag.name}: {child.text[:25]}"
                    )
                    print(self.content_name, self.content_category, self.url)
        return

    def _extract_text_from_em(self, tag: PageElement, content: list[str]) -> None:
        """
        Helper method to extract the text content from the em element.

        Args:
            tag (PageElement): The HTML element to extract the text elements from.
            content (list[str]): A list of text elements.

        Returns:
            None: This method modifies the content list in-place.

        Note:
            This method is used in `extract_text` method.
        """

        # FIXME: Remove `Image courtesy` or `Photo courtesy` string ('Photo courtesy' unresolved)
        for child in tag.children:
            if "Image Courtesy" in child.text or "Photo Courtesy" in child.text:
                continue
            elif child.name in ["a", "strong", "sup", "em"]:
                self._extract_text_elements(child, content)
            elif child.name is None:
                content.append(self.clean_text(child.text))
            elif child.name == "img":
                self._extract_text_elements(child, content)
            elif child.name == "span":
                self._extract_text_from_container(child, content)
            else:
                logger.debug(
                    f"Text Extraction - Tag {child.name} not handled within {tag.name}: {child.text[:25]}"
                )

    def _extract_text_from_ul(self, tag: PageElement, content: list[str]) -> None:
        """
        Helper method to extract the text content from the ul element.

        Args:
            tag (PageElement): The HTML element to extract the text elements from.
            content (list[str]): A list of text elements.

        Returns:
            None: This method modifies the content list in-place.

        Note:
            This method is used in `extract_text` method.
        """
        # Unwrap span tags within ul tags
        while tag.span is not None:
            tag.span.unwrap()

        # Unwrap div tags within ul tags
        while tag.div is not None:
            tag.div.unwrap()

        # Unwrap a tags within ul
        while tag.a is not None:
            tag.a.unwrap()

        # Unwrap strong tags within ul
        while tag.strong is not None:
            tag.strong.unwrap()

        for child in tag.children:
            if child.name == "li":
                # Unwrap span tags within li
                while not isinstance(child, NavigableString) and child.span is not None:
                    child.span.unwrap()
                # Unwrap a tags within li
                while not isinstance(child, NavigableString) and child.a is not None:
                    child.a.unwrap()
                while (
                    not isinstance(child, NavigableString) and child.strong is not None
                ):
                    child.strong.unwrap()
                content.append("- " + self.clean_text(child.text))
            elif child.name is None:
                cleaned_text = self.clean_text(child.text)
                if cleaned_text != "":
                    # TODO: Look into how to resolve nested text in bullets
                    # content.append("\t")
                    content.append(cleaned_text)
            elif child.name in ["p", "ul", "ol", "h2", "h3"]:
                cleaned_text = self.clean_text(child.text)
                if cleaned_text != "":
                    # TODO: Look into how to resolve nested text in bullets
                    # content.append("\t")
                    self._extract_text_elements(child, content)
            elif child.name == "img":
                self._extract_text_elements(child, content)
            else:
                cleaned_text = self.clean_text(child.text)
                logger.debug(
                    f"Text Extraction - Tag {child.name} not handled within {tag.name}: {cleaned_text[:25]}"
                )

        content.append("\n")

    def _extract_text_from_ol(self, tag: PageElement, content: list[str]) -> None:
        """
        Helper method to extract the text content from the ol element.

        Args:
            tag (PageElement): The HTML element to extract the text elements from.
            content (list[str]): A list of text elements.

        Returns:
            None: This method modifies the content list in-place.

        Note:
            This method is used in `extract_text` method.
        """
        # Unwrap span tags within ol
        while tag.span is not None:
            tag.span.unwrap()

        # Extract text from children
        start_counter = tag.get("start", 1)
        for i, child in enumerate(tag.children):
            if child.name == "li":
                content.append(
                    f"{int(start_counter) + i}. " + self.clean_text(child.text)
                )
            elif child.name is None:
                cleaned_text = self.clean_text(child.text)
                if cleaned_text != "":
                    # TODO: Look into how to resolve nested text in bullets
                    # content.append("\t")
                    content.append(cleaned_text)
            elif child.name in ["ul", "ol", "p", "h4"]:
                cleaned_text = self.clean_text(child.text)
                if cleaned_text != "":
                    # TODO: Look into how to resolve nested text in bullets
                    # content.append("\t")
                    self._extract_text_elements(child, content)
            else:
                logger.debug(
                    f"Text Extraction - Tag {child.name} not handled within {tag.name}: {child.text[:25]}"
                )

        content.append("\n")

    def _extract_text_from_container(
        self, tag: PageElement, content: list[str]
    ) -> None:
        """
        Helper method to extract text from div and span containers.

        This method recursively processes div, span and blockquote elements, extracting their text content.
        It handles both direct text content and nested elements.

        Args:
            tag (PageElement): The BeautifulSoup tag (div or span) to extract text from.
            content (list[str]): The list to append extracted text to.

        Returns:
            None: This method modifies the content list in-place.
        """
        # Check for text within its children
        for child in tag.children:
            # Recursive search
            if child.name in ["div", "span", "blockquote"]:
                self._extract_text_from_container(child, content)
            # Child has no HTML tag (i.e. texts in div containers)
            elif child.name is None:
                cleaned_text = self.clean_text(child.text)
                # Only extract text that are not solely punctuation
                if cleaned_text != "" and cleaned_text not in string.punctuation:
                    content.append(cleaned_text)
                # Add punctuation back to text
                elif cleaned_text in string.punctuation and len(content) > 0:
                    content[-1] = content[-1] + cleaned_text
                else:
                    cleaned_text = self.clean_text(child.text)
                    if cleaned_text != "":
                        # TODO: Monitor for edge cases & Resolve it
                        logger.debug(
                            f"Text Extraction - Text not handled within container - {child.name}: {cleaned_text[:25]}"
                        )
                        print(self.content_name, self.content_category, self.url)
            # Continue extracting text for other elements
            else:
                self._extract_text_elements(child, content)

    def check_for_table(self) -> bool:
        """
        Check for the presence of table tags in an HTML document.

        Returns:
            bool: True if at least one table tag is found, False otherwise.
        """
        # Find all table tags
        tables = self.soup.find_all("table")

        # Return True if at least one table is found, False otherwise
        return len(tables) > 0

    def check_for_image(self) -> bool:
        """
        Check for the presence of img tags in an HTML document.

        Returns:
            bool: True if at least one img tag is found, False otherwise.
        """
        # Find all img tags
        images = self.soup.find_all("img")

        # Return True if at least one image is found, False otherwise
        return len(images) > 0

    def extract_related_sections(self) -> list[str]:
        """
        Extracts "Related:" sections and "Read these next:" items from the HTML content.

        Returns:
            list[str]: A list of related sections and "Read these next:" items.
        """
        related_sections = []
        read_these_next_ul = None
        # Extract "Related:" sections and "Read these next:" items
        for tag in self.soup.find_all(["p", "ul"]):
            if tag.name == "p" and tag.find("strong"):
                cleaned_text = self.clean_text(tag.text)
                if "Related:" in cleaned_text:
                    related_sections.append(re.sub(r"Related: ", "", cleaned_text))
                elif "Read these next:" in cleaned_text:
                    read_these_next_ul = tag.find_next_sibling("ul")
            elif tag == read_these_next_ul:
                for li in tag.find_all("li"):
                    related_sections.append(self.clean_text(li.text))

        return related_sections

    def extract_links(self) -> list[tuple[str, str]]:
        """
        Extracts the title and URL from all the anchor tags in the HTML content.

        Returns:
            list[tuple[str, str]]:
                A list of tuples containing the title and URL of each anchor tag.

        Note:
            Footnotes to references sections are ignored.
        """
        extracted_links = []

        # Extract title/text and links from anchor tags
        for link in self.soup.find_all("a"):
            url = link.get("href")
            # Ignore footnotes
            if url != "#footnotes":
                text = link.get("title") or link.get_text()
                cleaned_text = self.clean_text(text)
                record = cleaned_text, url
                extracted_links.append(record)

        return extracted_links

    def extract_alt_text_from_img(self) -> list[str]:
        """
        Extracts the alternate text from images

        Returns:
            list[str]:
                A list of tuples containing the alternate text of images
        """
        # Note: In some articles, the img alternate text is the same as the header
        extracted_alt_text = []
        for img in self.soup.find_all("img"):
            alt_text = img.get("alt", None)
            if alt_text is None:
                continue
            cleaned_text = self.clean_text(alt_text)
            if cleaned_text != "":
                extracted_alt_text.append(cleaned_text)

        # Return unique elements
        return list(set(extracted_alt_text))

    def extract_headers(self) -> list[tuple[str, str]]:
        """
        Extracts the headers from the HTML content.

        Returns:
            list[tuple[str, str]]:
                A list of tuples containing the text and tag name of
                each header found in the HTML content.
        """
        extracted_headers = []

        for title in self.soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            tag = title.name
            text = self.clean_text(title.get_text())
            record = text, tag
            extracted_headers.append(record)

        return extracted_headers
